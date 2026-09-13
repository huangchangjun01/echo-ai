"""三维关系值加载与 prompt 渲染（role_relationship 表，M3 完整实现）。

设计要点:
- 从 role_relationship 表读 intimacy / trust / satisfaction / co_days
- render 控制在 60 token 以内(PRD 06 预算)
- 高亲密/信任给出"亲密关系"指令;低信任给出"保持专业"指令
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def load_relationship(user_id: str, role_id: str = "default") -> dict[str, Any] | None:
    """从 role_relationship 表加载三维关系值快照。

    无记录 / 失败 → 返回 None(由上层 fallback 走零值默认)。
    """
    if not user_id:
        return None
    try:
        from database import fetch_one
        from utils.request_context import log_exception

        row = await fetch_one(
            """
            SELECT intimacy, trust, satisfaction, co_days
            FROM role_relationship
            WHERE user_id=%s AND role_id=%s
            """,
            (user_id, role_id),
        )
        if not row:
            return None
        return {
            "intimacy": float(row.get("intimacy") or 0),
            "trust": float(row.get("trust") or 0),
            "satisfaction": float(row.get("satisfaction") or 0),
            "coDays": int(row.get("co_days") or 0),
        }
    except Exception as e:
        from utils.request_context import log_exception
        log_exception(
            logger,
            "load_relationship (role_relationship) failed",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.relationship",
            event="load_failed",
            user_id=user_id,
            role_id=role_id,
        )
        return None


def render_relationship_for_prompt(rel: dict[str, Any] | None) -> str:
    """三维关系值 → LLM 友好的关系策略提示词(<= 60 token)。

    根据亲密/信任/满意度给出:
    - 亲密关系模式(intimacy 高)
    - 信任程度(trust 高 → 更放松 / 低 → 保持专业)
    - 满意度(satisfaction 高 → 更主动 / 低 → 更谨慎)
    - 陪伴天数(社会证明,影响语气熟悉度)
    """
    if not rel:
        return ""
    intimacy = rel.get("intimacy", 0)
    trust = rel.get("trust", 0)
    satisfaction = rel.get("satisfaction", 0)
    co_days = rel.get("coDays", 0)

    # 亲密阈值:≥0.6 进入亲密模式
    intimacy_label = "亲密" if intimacy >= 0.6 else ("熟悉" if intimacy >= 0.3 else "陌生")
    # 信任阈值:≥0.6 信任,0.3~0.6 中等,<0.3 警惕
    trust_label = "高信任" if trust >= 0.6 else ("中性" if trust >= 0.3 else "低信任")
    # 满意度:≥0.6 主动,<0.3 谨慎
    sat_label = "主动" if satisfaction >= 0.6 else ("平稳" if satisfaction >= 0.3 else "谨慎")

    parts = [
        f"亲密度{intimacy:.2f}({intimacy_label})",
        f"信任{trust:.2f}({trust_label})",
        f"满意度{satisfaction:.2f}({sat_label})",
    ]
    if co_days > 0:
        parts.append(f"陪伴{co_days}天")
    return "【关系】" + " · ".join(parts)