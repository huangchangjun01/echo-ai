"""回忆记忆路由：/memory/parse, /memory/reparse, /memory/source/delete, /memory/delete。

挂在主 FastAPI app 上（保留原有 agent_runner.py 的 /chat、/ingest_file、/health 不动）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from biz.recall import delete_memory, delete_source, parse_memory, reparse_memory
from utils.request_context import (
    bind_request,
    log_exception,
    merge_extra,
    unbind_request,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class SourceFileIn(BaseModel):
    fileKey: str
    fileName: str
    fileType: int
    url: str | None = None


class ParseMemoryReq(BaseModel):
    userId: str
    roleId: str = "default"
    memoryId: str
    topic: str
    subjectiveDesc: str | None = None
    sourceFiles: list[SourceFileIn] = Field(default_factory=list)


@router.post("/memory/parse")
async def parse_memory_endpoint(req: ParseMemoryReq, background: BackgroundTasks) -> dict[str, Any]:
    """解析记忆（异步）：接收源文件列表 → 后台执行解析链路 → 立即返回。"""
    token = bind_request(stage="recall_parse", event="start", memory_id=req.memoryId, user_id=req.userId)
    try:
        sources = [s.model_dump() for s in req.sourceFiles]
        logger.info(
            "/memory/parse received",
            extra=merge_extra(
                stage="recall_parse",
                event="dispatch",
                user_id=req.userId,
                role_id=req.roleId,
                memory_id=req.memoryId,
                sources=len(sources),
                has_subjective=bool(req.subjectiveDesc),
            ),
        )

        async def _run() -> None:
            try:
                await parse_memory(
                    user_id=req.userId,
                    role_id=req.roleId,
                    memory_id=req.memoryId,
                    topic=req.topic,
                    subjective_desc=req.subjectiveDesc or "",
                    sources=sources,
                )
            except Exception as e:  # noqa: BLE001
                log_exception(
                    logger,
                    "/memory/parse background failed",
                    exc=e,
                    level=logging.WARNING,
                    stage="recall_parse",
                    event="bg_error",
                    memory_id=req.memoryId,
                )

        background.add_task(_run)
        return {"ok": True, "queued": True, "memoryId": req.memoryId}
    finally:
        unbind_request(token)


class DeleteSourceReq(BaseModel):
    userId: str
    roleId: str = "default"
    memoryId: str
    fileKey: str
    # fileName 客户端传过来（md 中按文件名匹配片段）
    fileName: str | None = None


@router.post("/memory/source/delete")
async def delete_source_endpoint(req: DeleteSourceReq, background: BackgroundTasks) -> dict[str, Any]:
    token = bind_request(stage="recall_delete_source", event="start", memory_id=req.memoryId)
    try:
        file_name = req.fileName or req.fileKey.split("/")[-1]
        logger.info(
            "/memory/source/delete received",
            extra=merge_extra(
                stage="recall_delete_source",
                event="dispatch",
                memory_id=req.memoryId,
                file_key=req.fileKey,
                file_name=file_name,
            ),
        )

        async def _run() -> None:
            try:
                await delete_source(req.userId, req.roleId, req.memoryId, req.fileKey, file_name)
            except Exception as e:  # noqa: BLE001
                log_exception(
                    logger,
                    "/memory/source/delete background failed",
                    exc=e,
                    level=logging.WARNING,
                    stage="recall_delete_source",
                    event="bg_error",
                )

        background.add_task(_run)
        return {"ok": True, "queued": True, "memoryId": req.memoryId}
    finally:
        unbind_request(token)


class DeleteMemoryReq(BaseModel):
    userId: str
    roleId: str = "default"
    memoryId: str


@router.post("/memory/delete")
async def delete_memory_endpoint(req: DeleteMemoryReq, background: BackgroundTasks) -> dict[str, Any]:
    token = bind_request(stage="recall_delete_memory", event="start", memory_id=req.memoryId)
    try:
        logger.info(
            "/memory/delete received",
            extra=merge_extra(stage="recall_delete_memory", event="dispatch", memory_id=req.memoryId),
        )

        async def _run() -> None:
            try:
                await delete_memory(req.userId, req.roleId, req.memoryId)
            except Exception as e:  # noqa: BLE001
                log_exception(
                    logger,
                    "/memory/delete background failed",
                    exc=e,
                    level=logging.WARNING,
                    stage="recall_delete_memory",
                    event="bg_error",
                )

        background.add_task(_run)
        return {"ok": True, "queued": True, "memoryId": req.memoryId}
    finally:
        unbind_request(token)


class ReparseMemoryReq(BaseModel):
    """编辑已有记忆后的再解析请求。

    与 ParseMemoryReq 的差异：
      - 多了 `removedFileKeys`：本次编辑中被软删的源文件 key 列表（仅用于日志/审计，
        实际"删片段、删对象、删向量"由 echo-core 事务前的 DeleteSourceFile 链路完成）。
      - `topic` 字段在本场景下其实不会被读取（parse_memory 生成的 md 标题由 skill
        `build_memory_md` 内部从 topic 取），保留它是为了与 ParseMemoryReq 形状对齐。
    """

    userId: str
    roleId: str = "default"
    memoryId: str
    topic: str = ""
    subjectiveDesc: str | None = None
    sourceFiles: list[SourceFileIn] = Field(default_factory=list)
    removedFileKeys: list[str] = Field(default_factory=list)


@router.post("/memory/reparse")
async def reparse_memory_endpoint(req: ReparseMemoryReq, background: BackgroundTasks) -> dict[str, Any]:
    """编辑后重新解析：先清旧向量 → 走标准解析链路（stage=`recall_reparse`）。

    与 /memory/parse 的关键差异在于日志 stage 与"先删向量"语义：
      - /memory/parse 用于"首次创建"，日志 stage=`recall_parse`
      - /memory/reparse 用于"编辑后再解析"，日志 stage=`recall_reparse`
    """
    token = bind_request(stage="recall_reparse", event="start", memory_id=req.memoryId)
    try:
        sources = [s.model_dump() for s in req.sourceFiles]
        logger.info(
            "/memory/reparse received",
            extra=merge_extra(
                stage="recall_reparse",
                event="dispatch",
                user_id=req.userId,
                role_id=req.roleId,
                memory_id=req.memoryId,
                sources=len(sources),
                removed=len(req.removedFileKeys),
            ),
        )

        async def _run() -> None:
            try:
                await reparse_memory(
                    user_id=req.userId,
                    role_id=req.roleId,
                    memory_id=req.memoryId,
                    topic=req.topic or "",
                    subjective_desc=req.subjectiveDesc or "",
                    sources=sources,
                    removed_file_keys=list(req.removedFileKeys),
                )
            except Exception as e:  # noqa: BLE001
                log_exception(
                    logger,
                    "/memory/reparse background failed",
                    exc=e,
                    level=logging.WARNING,
                    stage="recall_reparse",
                    event="bg_error",
                    memory_id=req.memoryId,
                )

        background.add_task(_run)
        return {"ok": True, "queued": True, "memoryId": req.memoryId}
    finally:
        unbind_request(token)