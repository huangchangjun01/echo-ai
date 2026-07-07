"""Echo-AI 入口：Agent 服务。

对外只暴露三个端点：
- GET  /health         存活/就绪探针
- POST /chat           Agent 对话（内部完成：persona 注入 + L0/L1 检索 +
                       轻量 ReAct + 工具调用 + 大小模型级联 + 异步记忆抽取）
- POST /ingest_file    文件入库（后台：下载 → 切片 → embedding → 写入 EchoDoc）

记忆、人格、工具、embedding 都属于 Agent 内部能力，不对外暴露。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from biz.chat import chat_collect, chat_stream
from biz.ingest import ingest_file
from config.config import get_settings
from database import close_pool as close_db_pool, init_schema
from embedding import bge_m3
from embedding.embeddings import ChineseCLIPEmbeddings
from utils.logging_setup import setup_logging
from utils.request_context import (
    bind_request,
    merge_extra,
    unbind_request,
)
from vector import vector_store as vs_module

# 入口处安装 JSON 日志设施（在路由装饰器之前）
setup_logging()
logger = logging.getLogger(__name__)


# ---------- Lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "Echo-AI starting",
        extra=merge_extra(
            stage="lifespan_start",
            event="start",
            version="2.0.0",
            threshold=settings.vector_similarity_threshold,
            llm_model=settings.llm.model,
            llm_base_url=settings.llm.base_url,
            db_host=settings.db.host,
            db_port=settings.db.port,
            db_name=settings.db.name,
            weaviate_url=settings.weaviate.resolved_url(),
            weaviate_class=settings.weaviate.class_name,
            memory_class=settings.weaviate.memory_class,
            small_model=settings.llm.small_model,
            bge_model=settings.bge_m3.model,
            embed_model=settings.embedding.model_name,
            embed_dim=settings.embedding.dim,
            bge_dim=settings.bge_m3.dim,
            host=settings.app.host,
            port=settings.app.port,
        ),
    )

    # 1) 建表（幂等）
    t0 = time.perf_counter()
    try:
        await init_schema()
        logger.info(
            "MySQL schema ready",
            extra=merge_extra(stage="init_schema", event="ok", duration_ms=round((time.perf_counter() - t0) * 1000, 2)),
        )
    except Exception as e:
        logger.exception(
            "MySQL init failed",
            extra=merge_extra(stage="init_schema", event="error", error=str(e)[:300]),
        )
        raise

    # 2) Embedding 预热（非阻塞）
    try:
        async def _warmup_bg() -> None:
            t1 = time.perf_counter()
            try:
                await asyncio.to_thread(bge_m3.warmup)
                logger.info(
                    "BGE-M3 warmup done",
                    extra=merge_extra(
                        stage="bge_warmup",
                        event="ok",
                        duration_ms=round((time.perf_counter() - t1) * 1000, 2),
                    ),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "BGE-M3 warmup failed (non-fatal)",
                    extra=merge_extra(
                        stage="bge_warmup",
                        event="error",
                        duration_ms=round((time.perf_counter() - t1) * 1000, 2),
                        error=str(e)[:300],
                    ),
                )

        app.state.warmup_task = asyncio.create_task(_warmup_bg())
        logger.info(
            "BGE-M3 warmup scheduled",
            extra=merge_extra(stage="bge_warmup", event="scheduled"),
        )
    except Exception as e:
        logger.warning(
            "BGE-M3 warmup schedule failed",
            extra=merge_extra(stage="bge_warmup", event="error", error=str(e)[:300]),
        )

    app.state.settings = settings
    logger.info(
        "Echo-AI agent ready",
        extra=merge_extra(stage="lifespan_start", event="end", ok=True),
    )
    try:
        yield
    finally:
        warmup_task = getattr(app.state, "warmup_task", None)
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
            try:
                await warmup_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await close_db_pool()
            logger.info("MySQL pool closed", extra=merge_extra(stage="lifespan_end", event="db_closed"))
        except Exception as e:
            logger.warning(
                "close_db_pool failed",
                extra=merge_extra(stage="lifespan_end", event="db_close_error", error=str(e)[:200]),
            )
        logger.info(
            "Echo-AI shutdown complete",
            extra=merge_extra(stage="lifespan_end", event="shutdown_ok"),
        )


app = FastAPI(
    title="Echo-AI",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------- 请求模型（仅对外暴露） ----------


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128, alias="userId")
    session_id: str | None = Field(None, alias="sessionId")
    message: str = Field(..., min_length=1, max_length=4096)
    stream: bool = False

    model_config = {"populate_by_name": True}


class FileObject(BaseModel):
    fileId: str
    fileName: str
    fileKey: str | None = None
    url: str | None = None


class IngestRequest(BaseModel):
    userId: str
    file: FileObject


# ---------- 路由 ----------


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {"status": "ok", "version": "2.0.0"}


@app.post("/chat")
async def chat_endpoint(req: ChatRequest) -> dict[str, Any]:
    """Agent 对话入口：内部完成 persona 注入、L0/L1 检索、ReAct、工具调用、
    大小模型级联、异步记忆抽取。

    请求体：`{userId, sessionId?, message, stream?}`
    响应：
    - `stream=false`：完整 JSON `{userId, sessionId, reply, events, latencyMs}`，且记忆抽取已落库
    - `stream=true`：SSE，逐帧 `data: {...}`（context / tool / prefix / delta / done），
      记忆抽取以 fire-and-forget 触发，避免阻塞 SSE
    """
    from memory import extract_and_archive_async

    user_id = req.user_id
    session_id = req.session_id or uuid.uuid4().hex
    request_id = uuid.uuid4().hex
    msg_len = len(req.message or "")
    ctx_token = bind_request(
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        stage="chat",
        event="start",
    )
    t0 = time.perf_counter()
    logger.info(
        "/chat received",
        extra=merge_extra(
            stream=bool(req.stream),
            msg_len=msg_len,
            msg_preview=(req.message or "")[:120],
        ),
    )
    try:
        if req.stream:
            async def _gen():
                nonlocal_full = {"v": ""}
                emitted = {"context": False, "prefix": False, "delta": False, "tool": 0}
                t = time.perf_counter()
                async for ev in chat_stream(user_id, session_id, req.message):
                    et = ev.get("type")
                    if et == "done":
                        nonlocal_full["v"] = ev.get("full", "") or ""
                    elif et == "context":
                        emitted["context"] = True
                    elif et == "prefix":
                        emitted["prefix"] = True
                    elif et == "delta":
                        emitted["delta"] = True
                    elif et == "tool":
                        emitted["tool"] += 1
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                logger.info(
                    "/chat sse done",
                    extra=merge_extra(
                        event="sse_end",
                        stream=True,
                        full_len=len(nonlocal_full["v"]),
                        full_preview=nonlocal_full["v"][:120],
                        sse_elapsed_ms=round((time.perf_counter() - t) * 1000, 2),
                        had_context=emitted["context"],
                        had_prefix=emitted["prefix"],
                        had_delta=emitted["delta"],
                        tool_events=emitted["tool"],
                    ),
                )
                # fire-and-forget 抽取：子任务里重新绑定上下文，确保 request_id 贯穿
                if nonlocal_full["v"]:
                    async def _extract():
                        inner_token = bind_request(
                            user_id=user_id,
                            session_id=session_id,
                            request_id=request_id,
                            stage="memory_extract",
                            event="background",
                        )
                        try:
                            await extract_and_archive_async(
                                user_id=user_id,
                                session_id=session_id,
                                user_msg=req.message,
                                assistant_msg=nonlocal_full["v"],
                            )
                        except Exception as e:  # noqa: BLE001
                            logger.warning(
                                "memory extract (fire-and-forget) failed",
                                extra=merge_extra(stage="memory_extract", event="error", error=str(e)[:300]),
                            )
                        finally:
                            unbind_request(inner_token)

                    asyncio.create_task(_extract())

            return StreamingResponse(_gen(), media_type="text/event-stream")

        result = await chat_collect(user_id, session_id, req.message)
        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "/chat done",
            extra=merge_extra(
                event="end",
                ok=True,
                stream=False,
                reply_len=len(result.get("full", "") or ""),
                reply_preview=(result.get("full", "") or "")[:120],
                events=len(result.get("events", []) or []),
                chat_latency_ms=result.get("latency_ms"),
                duration_ms=round(elapsed, 2),
            ),
        )
        return {
            "userId": user_id,
            "sessionId": session_id,
            "reply": result["full"],
            "events": result["events"],
            "latencyMs": result["latency_ms"],
        }
    except Exception as e:
        logger.exception(
            "/chat failed",
            extra=merge_extra(event="error", error=str(e)[:300]),
        )
        raise
    finally:
        unbind_request(ctx_token)


@app.post("/ingest_file")
async def ingest_file_endpoint(req: IngestRequest, background: BackgroundTasks) -> dict[str, Any]:
    """文件入库：后台异步执行（下载 → 切片 → embedding → 写入 EchoDoc）。
    仅用于把外部文档灌入知识库；与 /chat 内的记忆/人格/工具能力无关。
    """
    request_id = uuid.uuid4().hex
    t0 = time.perf_counter()
    file_dict = req.file.model_dump()

    logger.info(
        "/ingest_file received",
        extra=merge_extra(
            stage="ingest_file",
            event="start",
            user_id=req.userId,
            request_id=request_id,
            fileId=req.file.fileId,
            fileName=req.file.fileName,
            has_fileKey=bool(req.file.fileKey),
            has_url=bool(req.file.url),
            url=req.file.url or "",
        ),
    )

    embeddings = ChineseCLIPEmbeddings()
    try:
        vectorstore = await asyncio.to_thread(vs_module.get_vector_store)
    except Exception as e:
        logger.exception(
            "vector store init failed",
            extra=merge_extra(stage="ingest_file", event="error", error=str(e)[:300]),
        )
        raise HTTPException(status_code=500, detail=f"vector store init failed: {e}")

    queue_elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        "/ingest_file queued",
        extra=merge_extra(stage="ingest_file", event="dispatch", queue_elapsed_ms=queue_elapsed_ms),
    )

    # FastAPI 的 BackgroundTasks.add_task 支持 async 函数；
    # 这里把 ctx_token 一起通过闭包传给任务函数，让 ingest_file 阶段也带上 request_id。
    background.add_task(
        _run_ingest_with_ctx,
        ctx_snapshot_factory=lambda: {
            "user_id": req.userId,
            "request_id": request_id,
            "stage": "ingest_file",
        },
        ingest_func=ingest_file,
        user_id=req.userId,
        file_obj=file_dict,
        embeddings=embeddings,
        vectorstore=vectorstore,
    )
    return {"ok": True, "queued": True, "fileId": req.file.fileId}


# ---------- BackgroundTasks 辅助 ----------

async def _run_ingest_with_ctx(
        *,
        ctx_snapshot_factory,
        ingest_func,
        **kwargs,
):
    """在后台任务里先绑请求上下文，再跑 ingest。

    注：BackgroundTasks 不会传播外层 `bind_request` 的 ContextVar；
    这里把上下文快照从外层带进，并在任务里重新绑定。
    """
    token = bind_request(**ctx_snapshot_factory())
    try:
        await ingest_func(**kwargs)
    finally:
        unbind_request(token)
