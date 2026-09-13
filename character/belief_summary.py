"""信念条目加载与 prompt 摘要(M3 完整实现)。

设计要点:
- 读 role_belief 表 user_confirmed=1 的"已确认"信念
- 按 confidence 降序,最多 10 条
- 进程内 5 分钟 LRU 缓存(PRD 06 预算 150 token,缓存粒度 user)
- render 控制总长 ≤ 150 token
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_TTL_SEC = 300  # 5 分钟
_BELIEF_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}


async def load_belief_summary(user_id: str, role_id: str = "default") -> list[dict[str, Any]]:
    """加载当前角色已确认信念(<= 10 条,confidence 降序)。

    带 5 分钟 LRU 缓存(单用户级别),避免每轮对话都查 DB。
    """
    if not user_id:
        return []
    cache_key = f"{user_id}:{role_id}"
    cached = _BELIEF_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SEC:
        return cached[1]

    try:
        from database import fetch_all
        from utils.request_context import log_exception

        rows = await fetch_all(
            """
            SELECT topic, stance, confidence, evidence_count
            FROM role_belief
            WHERE user_id=%s AND role_id=%s AND user_confirmed=1
            ORDER BY confidence DESC, id DESC
            LIMIT 10
            """,
            (user_id, role_id),
        )
        items = list(rows or [])
        _BELIEF_CACHE[cache_key] = (time.time(), items)
        return items
    except Exception as e:
        from utils.request_context import log_exception
        log_exception(
            logger,
            "load_belief_summary (role_belief) failed",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.belief_summary",
            event="load_failed",
            user_id=user_id,
            role_id=role_id,
        )
        return []


def render_belief_summary_for_prompt(beliefs: list[dict[str, Any]]) -> str:
    """信念列表 → LLM 友好的摘要(≤ 150 token)。

    格式:
      【信念立场】
      - topic(置信度 X%): stance
      ...
    """
    if not beliefs:
        return ""

    lines = ["【信念立场】"]
    for b in beliefs[:10]:
        topic = (b.get("topic") or "").strip()[:30]
        stance = (b.get("stance") or "").strip()[:80]
        try:
            conf = float(b.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0
        pct = round(conf * 100)
        # 置信度低的信念用"倾向"措辞,高的用"坚持"
        verb = "坚持" if pct >= 70 else ("倾向" if pct >= 40 else "可能有")
        if stance:
            lines.append(f"- {verb}{topic}(置信{pct}%):{stance}")
        else:
            lines.append(f"- {verb}{topic}(置信{pct}%)")
    return "\n".join(lines)