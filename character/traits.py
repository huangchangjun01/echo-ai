"""OCEAN 五维性格加载(role_traits 表)。

设计要点:
- 优先读 role_traits 表(复合主键 user_id+role_id)
- 若表不存在 / 无记录 -> 返回 None(由上层 fallback 到预设基线 + DEFAULT_PERSONA)
- 严格只读,不写主表
"""

from __future__ import annotations

import logging
from typing import Any

from utils.request_context import log_exception

from .presets import get_preset_traits

logger = logging.getLogger(__name__)


async def load_traits(user_id: str, role_id: str = "default") -> dict[str, Any] | None:
    """从 role_traits 表加载 OCEAN 五维。

    Args:
        user_id: 用户 ID
        role_id: 角色 ID

    Returns:
        dict 含 openness/conscientiousness/extraversion/agreeableness/neuroticism/presetType
        若表不存在 / 无记录 / 失败,返回 None
    """
    if not user_id:
        return None
    try:
        from database import fetch_one

        row = await fetch_one(
            """
            SELECT openness, conscientiousness, extraversion, agreeableness,
                   neuroticism, preset_type
            FROM role_traits
            WHERE user_id=%s AND role_id=%s
            """,
            (user_id, role_id),
        )
        if not row:
            return None
        return {
            "openness": float(row.get("openness") or 0),
            "conscientiousness": float(row.get("conscientiousness") or 0),
            "extraversion": float(row.get("extraversion") or 0),
            "agreeableness": float(row.get("agreeableness") or 0),
            "neuroticism": float(row.get("neuroticism") or 0),
            "presetType": row.get("preset_type") or "companion_default",
        }
    except Exception as e:
        log_exception(
            logger,
            "load_traits (role_traits) failed, will fallback",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.traits",
            event="load_failed",
            user_id=user_id,
            role_id=role_id,
        )
        return None


def render_traits_for_prompt(
    traits: dict[str, Any] | None, preset_type: str | None = None
) -> str:
    """把 OCEAN 五维渲染成 LLM 能直接采纳的提示词片段(<= 80 token)。

    设计要点:
    1. 不仅是数值,还把每维 OCEAN 拆成"档位标签 + 行为指令",让 LLM 知道
       "该用 OCEAN 干什么",而不只是看到一串数字不知道含义。
    2. 档位分 5 档:极高 / 高 / 中 / 低 / 极低(每档对应具体的行为倾向)。
    3. 末尾给一条全局指令:"你的回复必须体现以下性格倾向",让 LLM 真正
       把 OCEAN 应用到回复内容(而不只是心情基线)。
    """
    if traits is None:
        traits_dict = dict(get_preset_traits(preset_type or "companion_default"))
        preset_type = preset_type or "companion_default"
    else:
        traits_dict = traits

    def level(v: float, direction: str) -> str:
        """把 [-1, 1] 数值映射成 5 档标签。

        direction 是预定义的英文/数字短串,避免任何 f-string/Unicode 解析问题。
        """
        if v >= 0.6:   return f"V+ {direction}"
        if v >= 0.2:   return f"V+ {direction}"
        if v > -0.2:   return f"~0 {direction}"
        if v > -0.6:   return f"V- {direction}"
        return f"V- {direction}"

    o = traits_dict["openness"]
    c = traits_dict["conscientiousness"]
    e = traits_dict["extraversion"]
    a = traits_dict["agreeableness"]
    n = traits_dict["neuroticism"]

    return (
        "[OCEAN, -1..+1, MUST apply in tone/word choice/initiative]\n"
        f"O open {o:+.2f} {level(o, 'curious, try new')}; "
        f"C cons {c:+.2f} {level(c, 'organized, reliable')}; "
        f"E extr {e:+.2f} {level(e, 'outgoing, initiate')}; "
        f"A agre {a:+.2f} {level(a, 'warm, empathetic')}; "
        f"N neur {n:+.2f} {level(n, 'anxious OR calm-stable')}.\n"
        "Match these in every reply (especially N)."
    )