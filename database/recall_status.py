"""recall_memory 表的轻量读写（echo-ai 与 echo-core 共享同一 MySQL）。

仅做 echo-ai 视角需要的字段更新：parse_status / edit_status / md_key。
不做业务校验（userId、topic 唯一性由 echo-core 保证）。
"""

from __future__ import annotations

import logging

from database import execute, fetch_one
from utils.request_context import log_exception

logger = logging.getLogger(__name__)


async def update_status(
    memory_id: str,
    *,
    parse_status: int | None = None,
    edit_status: int | None = None,
) -> None:
    fields: list[str] = []
    params: list = []
    if parse_status is not None:
        fields.append("parse_status = %s")
        params.append(parse_status)
    if edit_status is not None:
        fields.append("edit_status = %s")
        params.append(edit_status)
    if not fields:
        return
    params.append(memory_id)
    sql = f"UPDATE recall_memory SET {', '.join(fields)} WHERE memory_id = %s"
    try:
        await execute(sql, tuple(params))
    except Exception as e:
        log_exception(
            logger,
            "update recall_memory status failed",
            exc=e,
            level=logging.WARNING,
            stage="recall_db",
            event="update_error",
            memory_id=memory_id,
        )


async def try_acquire_edit_lock(memory_id: str, max_wait_seconds: float = 60.0) -> bool:
    """尝试将 edit_status 设为 1（AI 写入中）。

    若当前已是 1，等待并轮询直到对方释放或超时。
    返回 True=拿到锁，False=超时失败。
    """
    import asyncio
    import time

    deadline = time.monotonic() + max_wait_seconds
    backoff = 0.5
    while True:
        row = await fetch_one(
            "SELECT edit_status FROM recall_memory WHERE memory_id = %s",
            (memory_id,),
        )
        if row is None:
            return False
        if int(row.get("edit_status") or 0) == 0:
            # 抢锁：CAS，用条件 update
            sql = "UPDATE recall_memory SET edit_status = 1 WHERE memory_id = %s AND edit_status = 0"
            n = await execute(sql, (memory_id,))
            if n > 0:
                return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(backoff)
        backoff = min(backoff * 1.5, 3.0)


async def release_edit_lock(memory_id: str) -> None:
    await update_status(memory_id, edit_status=0)


async def mark_done(memory_id: str, *, parse_status: int) -> None:
    await update_status(memory_id, parse_status=parse_status, edit_status=0)