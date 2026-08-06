"""对话记忆：长期会话表 + 短期消息缓冲 + 遗忘机制。

注意：本模块与回忆记忆完全隔离。回忆记忆永不遗忘；对话记忆有遗忘机制。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from database import execute, fetch_all, fetch_one
from utils.request_context import log_exception, merge_extra

logger = logging.getLogger(__name__)


# ===== 会话摘要 =====

async def upsert_session(
    user_id: str,
    role_id: str,
    session_id: str,
    *,
    msg_count_delta: int = 1,
    importance: float | None = None,
    summary: str | None = None,
) -> None:
    """会话自增：每次新消息 +1，重置 last_msg_at，可选更新 summary。"""
    role_id = role_id or "default"
    # 用 ON DUPLICATE KEY UPDATE 自增 + 触碰 last_msg_at
    sql = """
        INSERT INTO chat_sessions (user_id, role_id, session_id, msg_count, importance, summary, last_msg_at)
        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON DUPLICATE KEY UPDATE
            msg_count = msg_count + VALUES(msg_count),
            last_msg_at = CURRENT_TIMESTAMP,
            importance = COALESCE(VALUES(importance), importance),
            summary = COALESCE(VALUES(summary), summary)
    """
    try:
        await execute(sql, (user_id, role_id, session_id, msg_count_delta, importance or 0.3, summary))
    except Exception as e:
        log_exception(
            logger,
            "upsert_session failed",
            exc=e,
            level=logging.WARNING,
            stage="chat_memory",
            event="session_upsert_error",
            session_id=session_id,
        )


# ===== 消息追加 =====

async def append_message(
    session_id: str,
    user_id: str,
    role_id: str,
    role: str,
    content: str,
    *,
    token_count: int = 0,
) -> None:
    sql = """
        INSERT INTO chat_messages (session_id, user_id, role_id, role, content, token_count)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    try:
        await execute(sql, (session_id, user_id, role_id or "default", role, content, token_count))
    except Exception as e:
        log_exception(
            logger,
            "append_message failed",
            exc=e,
            level=logging.WARNING,
            stage="chat_memory",
            event="msg_append_error",
            session_id=session_id,
        )


# ===== 读取最近 N 轮原始消息（短期上下文） =====

async def recent_messages(session_id: str, limit: int = 10) -> list[dict[str, Any]]:
    sql = """
        SELECT role, content, created_at FROM chat_messages
        WHERE session_id = %s ORDER BY id DESC LIMIT %s
    """
    try:
        rows = await fetch_all(sql, (session_id, limit))
        # 倒序取回 → 调整为正序（按时间从旧到新）
        rows.reverse()
        return rows
    except Exception as e:
        log_exception(
            logger,
            "recent_messages failed",
            exc=e,
            level=logging.WARNING,
            stage="chat_memory",
            event="recent_msg_error",
            session_id=session_id,
        )
        return []


# ===== 遗忘机制 =====

async def gc_forgotten_sessions(max_age_days: int = 30, max_keep_per_user: int = 50) -> int:
    """定期 GC：
    1. 删除 last_msg_at 超过 max_age_days 天 且 retain_score < 0.05 的会话（消息级联删）
    2. 每个用户最多保留 max_keep_per_user 个最活跃会话
    """
    n = 0
    # 1) 时间衰减 + 访问频率阈值
    sql_age = """
        DELETE FROM chat_sessions
        WHERE last_msg_at < (NOW() - INTERVAL %s DAY)
          AND retain_score < 0.05
    """
    try:
        await execute(sql_age, (max_age_days,))
        n += 1
    except Exception as e:
        log_exception(logger, "gc age delete failed", exc=e, level=logging.WARNING, stage="chat_memory_gc", event="age_error")
    # 2) 单用户过量：按 retain_score 倒序保留前 max_keep_per_user 条，剩余标记 archived
    sql_trim = """
        UPDATE chat_sessions SET archived = 1, retain_score = 0.0
        WHERE id IN (
          SELECT id FROM (
            SELECT id, ROW_NUMBER() OVER (PARTITION BY user_id, role_id ORDER BY retain_score DESC, last_msg_at DESC) AS rn
            FROM chat_sessions WHERE archived = 0
          ) t WHERE rn > %s
        )
    """
    try:
        await execute(sql_trim, (max_keep_per_user,))
        n += 1
    except Exception as e:
        log_exception(logger, "gc trim failed", exc=e, level=logging.WARNING, stage="chat_memory_gc", event="trim_error")
    logger.info("chat_memory gc done", extra=merge_extra(stage="chat_memory_gc", event="ok", touched_tables=n))
    return n


async def decay_retain_scores(half_life_days: float = 14.0) -> int:
    """周期性按指数衰减 retain_score（半衰期语义）。"""
    import math
    sql = """
        UPDATE chat_sessions
        SET retain_score = retain_score * POW(0.5, (TIMESTAMPDIFF(MINUTE, last_msg_at, NOW()) / 1440.0) / %s)
        WHERE archived = 0
    """
    try:
        await execute(sql, (half_life_days,))
        return 1
    except Exception as e:
        log_exception(logger, "decay failed", exc=e, level=logging.WARNING, stage="chat_memory_gc", event="decay_error")
        return 0


async def bump_retain(session_id: str, delta: float = 0.2) -> None:
    """访问命中时上调 retain_score（防 GC）。"""
    sql = "UPDATE chat_sessions SET retain_score = LEAST(retain_score + %s, 5.0) WHERE session_id = %s"
    try:
        await execute(sql, (delta, session_id))
    except Exception as e:
        log_exception(logger, "bump_retain failed", exc=e, level=logging.DEBUG, stage="chat_memory", event="bump_error")


def start_gc_loop(interval_seconds: float = 6 * 3600) -> None:
    """由 FastAPI lifespan 调起：在主事件循环中跑 GC 循环（共享连接池）。

    注：绝不能在独立线程跑（aiomysql pool 绑定主 loop，会报
    "got Future ... attached to a different loop"）。
    """
    import asyncio

    async def _loop() -> None:
        logger.info("chat memory GC loop started | interval=%.0fs", interval_seconds)
        while True:
            try:
                await decay_retain_scores()
                await gc_forgotten_sessions()
            except Exception as e:
                logger.debug("gc loop tick failed (ignored): %s", e)
            await asyncio.sleep(interval_seconds)

    # 提交到主事件循环
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_loop())
    except RuntimeError:
        # 不在异步上下文（lifespan 启动失败），静默
        logger.warning("start_gc_loop: no running loop, GC disabled")