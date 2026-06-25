from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from biz.ingest import ingest_file
from config.config import get_settings
from embedding.embeddings import ChineseCLIPEmbeddings
from vector import vector_store as vs_module

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ── 可选依赖导入 ──────────────────────────────────────────────────
_MEMORY_AVAILABLE = False
_LLM_AVAILABLE = False
_TOOLS_AVAILABLE = False

try:
    from memory.retrieve import retrieve_l0_memories  # noqa: F401

    _MEMORY_AVAILABLE = True
except ImportError as e:
    logger.warning("memory module not available: %s", e)

try:
    from memory.extract import extract_and_archive  # noqa: F401
except ImportError:
    # extract_and_archive 可能独立不可用，但 memory 已可用时不影响
    if _MEMORY_AVAILABLE:
        logger.warning("memory.extract module not available")

try:
    from llm.react import (
        ReActLoop,
        build_tool_definitions as _build_react_tool_defs,
        register_tool as _register_react_tool,
    )

    _LLM_AVAILABLE = True
except ImportError as e:
    logger.warning("llm module not available: %s", e)

try:
    import tools  # noqa: F401  # 触发 tools/__init__.py 中的自动注册

    from tools import TOOL_REGISTRY as _TOOLS_REGISTRY, build_tool_definitions as _build_tools_defs

    _TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning("tools module not available: %s", e)


# ── 工具注册辅助函数 ──────────────────────────────────────────────


def _register_all_tools() -> None:
    """将 tools 模块中的工具注册到 llm.react 的 _TOOL_REGISTRY 中。"""
    if not _TOOLS_AVAILABLE or not _LLM_AVAILABLE:
        return
    for name, fn in _TOOLS_REGISTRY.items():
        try:
            _register_react_tool(name, fn)
        except Exception as e:
            logger.warning("Failed to register tool %s: %s", name, e)


def _build_combined_tool_definitions() -> list[dict[str, Any]]:
    """构建合并后的工具定义列表（vector_search + 自定义工具）。"""
    definitions: list[dict[str, Any]] = []
    if _LLM_AVAILABLE:
        try:
            definitions.extend(_build_react_tool_defs())
        except Exception as e:
            logger.warning("Failed to build react tool definitions: %s", e)
    if _TOOLS_AVAILABLE:
        try:
            definitions.extend(_build_tools_defs())
        except Exception as e:
            logger.warning("Failed to build tools definitions: %s", e)
    return definitions


def _format_l0_memories(memories: list[dict[str, Any]]) -> str:
    """将 L0 记忆列表格式化为用于系统提示词的文本。"""
    if not memories:
        return ""
    lines: list[str] = []
    for mem in memories:
        content = mem.get("content", "")
        importance = mem.get("importance", 0.0)
        emotion = mem.get("emotion_tag", "")
        line = f"- [{emotion}] (重要性: {importance:.2f}) {content}" if emotion else f"- (重要性: {importance:.2f}) {content}"
        lines.append(line)
    return "\n".join(lines)


# ── 生命周期 ──────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Echo-AI (threshold=%.3f)", settings.vector_similarity_threshold)

    # 1. 初始化 MySQL 连接池（含超时保护）
    try:
        from database.mysql import init_db

        await asyncio.wait_for(init_db(), timeout=5.0)
        app.state.mysql_available = True
        logger.info("MySQL connection pool initialized")
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning("MySQL init failed (non-fatal): %s", e)
        app.state.mysql_available = False

    # 2. 初始化工具注册
    try:
        _register_all_tools()
        logger.info("Tools registered in ReAct loop")
    except Exception as e:
        logger.warning("Tool registration failed: %s", e)

    # 3. Embedding 预热
    embeddings = ChineseCLIPEmbeddings()
    if settings.embedding.warmup_on_start:
        await asyncio.to_thread(embeddings.warmup)

    # 4. Vector store 初始化
    try:
        vectorstore = await asyncio.to_thread(vs_module.get_vector_store)
    except Exception as e:
        logger.error("Vector store init failed: %s", e)
        raise

    app.state.embeddings = embeddings
    app.state.vectorstore = vectorstore
    app.state.settings = settings
    logger.info("Echo-AI ready")

    try:
        yield
    finally:
        # 释放 MySQL 连接池
        try:
            from database.mysql import close_db

            await close_db()
            logger.info("MySQL connection pool released")
        except Exception:
            pass

        # 释放 Weaviate 连接
        try:
            await asyncio.to_thread(vs_module.reset_vector_store)
        except Exception:
            pass

        logger.info("Echo-AI shutdown complete")


app = FastAPI(
    title="Echo-AI Agent",
    version="2.0.0",
    lifespan=lifespan,
)


# ── 原有 Pydantic 模型 ────────────────────────────────────────────


class EmbeddingsRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=256)


class AddRequest(BaseModel):
    ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, Any]] | None = None


class FileObject(BaseModel):
    fileId: str
    fileName: str
    fileKey: str | None = None
    url: str | None = None


class IngestRequest(BaseModel):
    userId: str
    file: FileObject


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)
    k: int = Field(5, ge=1, le=50)
    user_id: str = Field(..., min_length=1, max_length=128, alias="userId")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)
    user_id: str = Field(..., min_length=1, max_length=128, alias="userId")


# ── 新增 Pydantic 模型 ────────────────────────────────────────────


class ChatCompletionRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1, max_length=128, alias="userId")
    stream: bool = False
    k: int = Field(5, ge=1, le=50)


class ChatCompletionResponse(BaseModel):
    id: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any]


class MemoryStatusRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128, alias="userId")


# ── 辅助函数 ──────────────────────────────────────────────────────


def _embedding_fn_for(embeddings: ChineseCLIPEmbeddings):
    def _fn(texts: list[str]) -> list[list[float]]:
        return embeddings.embed_documents(texts)

    return _fn


# ── 原有接口 ──────────────────────────────────────────────────────


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    embeddings = request.app.state.embeddings
    vectorstore = request.app.state.vectorstore
    where = {"userId": req.user_id}
    try:
        result = await asyncio.to_thread(
            vectorstore.query,
            req.query,
            req.k,
            _embedding_fn_for(embeddings),
            where,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(e))

    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    mds = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    candidates = [
        {"id": id_, "document": doc, "metadata": md, "distance": d}
        for id_, doc, md, d in zip(ids, docs, mds, distances)
    ]
    return {"query": req.query, "k": req.k, "userId": req.user_id, "candidates": candidates}


@app.post("/search")
async def search(req: SearchRequest, request: Request):
    return await chat(req=ChatRequest(query=req.query, k=req.k, userId=req.user_id), request=request)


@app.post("/ingest_file")
async def ingest_file_endpoint(req: IngestRequest, background: BackgroundTasks, request: Request):
    embeddings = request.app.state.embeddings
    vectorstore = request.app.state.vectorstore
    background.add_task(ingest_file, req.userId, req.file.model_dump(), embeddings, vectorstore)
    return {"ok": True, "queued": True, "fileId": req.file.fileId}


@app.get("/health")
async def health_check():
    """Lightweight health endpoint so the service can be probed by uptime checks."""
    return {"status": "ok"}


# ── 新增接口：POST /v1/chat/completions ───────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, background: BackgroundTasks, request: Request):
    """OpenAI 兼容的对话补全接口，支持 ReAct 循环和流式输出。"""

    # 检查依赖可用性
    if not _LLM_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="LLM service not available. Please configure LLM settings (LARGE_LLM_*).",
        )
    if not _MEMORY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Memory service not available. Please configure database (DB_*).",
        )

    user_id = req.user_id

    # a. 预注入：从 memory.retrieve 获取 L0 核心记忆
    memory_context = ""
    try:
        l0_memories = await retrieve_l0_memories(user_id)
        memory_context = _format_l0_memories(l0_memories)
    except Exception as e:
        logger.warning("Failed to retrieve L0 memories for user=%s: %s", user_id, e)

    # b. 构建工具定义
    tool_definitions = _build_combined_tool_definitions()

    # c. 构建 ReAct 循环
    react = ReActLoop(
        messages=req.messages,
        tools=tool_definitions if tool_definitions else None,
        memory_context=memory_context,
        max_iterations=5,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if req.stream:
        # d. 流式 SSE 输出
        async def _event_stream() -> AsyncGenerator[str, None]:
            collected_content: list[str] = []
            try:
                async for token in react.run_stream():
                    collected_content.append(token)
                    sse_data = json.dumps({"token": token}, ensure_ascii=False)
                    yield f"data: {sse_data}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.exception("ReAct stream failed for user=%s", user_id)
                error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                # e. 异步记忆抽取和归档
                full_content = "".join(collected_content)
                if full_content:
                    _schedule_memory_extraction(user_id, req.messages, full_content, background)

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # d. 非流式返回完整回复
        try:
            reply_content = await react.run()
        except Exception as e:
            logger.exception("ReAct run failed for user=%s", user_id)
            raise HTTPException(status_code=500, detail=f"LLM inference failed: {e}")

        # e. 异步记忆抽取和归档
        _schedule_memory_extraction(user_id, req.messages, reply_content, background)

        return ChatCompletionResponse(
            id=completion_id,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": reply_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )


def _schedule_memory_extraction(
    user_id: str,
    messages: list[dict[str, Any]],
    reply_content: str,
    background: BackgroundTasks,
) -> None:
    """异步调度记忆抽取和分层归档。"""
    try:
        # 构建完整的对话消息列表（包含 assistant 回复）
        full_messages = [
            *messages,
            {"role": "assistant", "content": reply_content},
        ]

        async def _extract():
            try:
                from memory.extract import extract_and_archive

                result = await extract_and_archive(user_id, full_messages)
                logger.info(
                    "Memory extraction completed for user=%s: archived=%d relations=%d",
                    user_id,
                    result.get("archived_count", 0),
                    result.get("relations_count", 0),
                )
            except Exception as e:
                logger.warning("Memory extraction failed for user=%s: %s", user_id, e)

        background.add_task(_extract)
    except Exception as e:
        logger.warning("Failed to schedule memory extraction for user=%s: %s", user_id, e)


# ── 新增接口：POST /v1/memory/status ──────────────────────────────


@app.post("/v1/memory/status")
async def memory_status(req: MemoryStatusRequest, request: Request):
    """查询用户各层记忆数量和最新摘要（调试用）。"""
    user_id = req.user_id

    if not getattr(request.app.state, "mysql_available", False):
        raise HTTPException(
            status_code=503,
            detail="MySQL not available. Please configure database (DB_*).",
        )

    try:
        from sqlalchemy import func, select

        from database.models import CoreMemory, MemorySummary
        from database.mysql import get_session

        l0_count = 0
        l1_count = 0
        l2_count = 0
        latest_summary: str | None = None

        async with get_session() as session:
            # 按层统计记忆数量
            result = await session.execute(
                select(CoreMemory.layer, func.count(CoreMemory.id))
                .where(CoreMemory.user_id == user_id)
                .group_by(CoreMemory.layer)
            )
            for layer, count in result.all():
                if layer == "L0":
                    l0_count = count
                elif layer == "L1":
                    l1_count = count
                elif layer == "L2":
                    l2_count = count

            # 获取最新摘要
            summary_result = await session.execute(
                select(MemorySummary)
                .where(MemorySummary.user_id == user_id)
                .order_by(MemorySummary.created_at.desc())
                .limit(1)
            )
            summary_row = summary_result.scalars().first()
            if summary_row:
                latest_summary = summary_row.summary

        return {
            "user_id": user_id,
            "l0_count": l0_count,
            "l1_count": l1_count,
            "l2_count": l2_count,
            "latest_summary": latest_summary,
        }
    except Exception as e:
        logger.exception("Failed to get memory status for user=%s", user_id)
        raise HTTPException(status_code=500, detail=str(e))