"""结构化人格加载（role_persona 表 + personas 表兼容）。

设计要点（PRD 02_PRD_data_model.md + 历史教训 842d7c5）：
- 优先读 role_persona 表（复合主键 user_id+role_id），新版本角色内核的主表
- 若 role_persona 表不存在 / 无记录 → 降级读 personas 表（旧结构，user_id 单主键）
- 若都无 → 返回 DEFAULT_PERSONA
- 严格只读，不写主表（PRD 铁律 #1 — LLM 抽取只写 suggestions 表）
"""

from __future__ import annotations

import json
import logging
from typing import Any

from utils.request_context import log_exception
from utils.request_context import current_context, merge_extra

from .presets import get_preset_name

logger = logging.getLogger(__name__)


async def load_persona(user_id: str, role_id: str = "default") -> dict[str, Any] | None:
    """从 role_persona 表加载结构化人格。

    Args:
        user_id: 用户 ID
        role_id: 角色 ID（默认 'default'）

    Returns:
        dict 含 identity / background / traitsTags / values / speakingStyle / taboos / exampleDialogs
        若表不存在 / 无记录 / 查询失败，返回 None（由上层 fallback）
    """
    if not user_id:
        return None
    try:
        from database import fetch_one

        row = await fetch_one(
            """
            SELECT identity, background, traits_tags, `values`, speaking_style,
                   taboos, example_dialogs, version
            FROM role_persona
            WHERE user_id=%s AND role_id=%s
            """,
            (user_id, role_id),
        )
        if not row:
            return None
        return {
            "identity": row.get("identity") or "",
            "background": row.get("background") or "",
            "traitsTags": _safe_json_list(row.get("traits_tags")),
            "values": _safe_json_list(row.get("values")),
            "speakingStyle": row.get("speaking_style") or "",
            "taboos": _safe_json_list(row.get("taboos")),
            "exampleDialogs": _safe_json_list(row.get("example_dialogs")),
            "version": row.get("version") or 1,
        }
    except Exception as e:
        log_exception(
            logger,
            "load_persona (role_persona) failed, will fallback",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.persona",
            event="load_failed",
            user_id=user_id,
            role_id=role_id,
        )
        return None


async def load_persona_legacy(user_id: str) -> str:
    """从 personas 表（旧结构）加载人格字符串。与历史 persona 读取路径完全兼容。

    Returns:
        persona 字符串（与 personas.persona 字段一致）
        若表不存在/无记录/失败，返回 DEFAULT_PERSONA
    """
    from config.prompts import DEFAULT_PERSONA

    if not user_id:
        return DEFAULT_PERSONA
    try:
        from database import fetch_one

        row = await fetch_one("SELECT persona FROM personas WHERE user_id=%s", (user_id,))
        if row and row.get("persona"):
            return row["persona"]
    except Exception as e:
        log_exception(
            logger,
            "load_persona_legacy failed",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.persona",
            event="legacy_load_failed",
            user_id=user_id,
        )
    return DEFAULT_PERSONA


async def load_persona_segment(user_id: str, role_id: str = "default") -> str:
    """返回「实际注入 LLM 的人格段」：role_persona → 旧 personas 表 → DEFAULT_PERSONA。

    与 build_segments 的 persona 段共用同一降级链（见 prompts_inject.py），
    让前端胶囊展示的人格与 LLM 实际看到的人格保持一致（角色级优先）。
    """
    from config.prompts import DEFAULT_PERSONA

    try:
        new_persona = await load_persona(user_id, role_id)
        if new_persona is not None:
            return render_persona_for_prompt(new_persona)
        # 降级：旧 personas 表（无记录时函数内部已返回 DEFAULT_PERSONA）
        return await load_persona_legacy(user_id)
    except Exception as e:
        log_exception(
            logger,
            "load_persona_segment failed, fallback to DEFAULT_PERSONA",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.persona",
            event="segment_failed",
            user_id=user_id,
            role_id=role_id,
        )
        return DEFAULT_PERSONA


def render_persona_for_prompt(persona: dict[str, Any] | None) -> str:
    """把结构化人格渲染成 LLM 友好的提示词片段。

    设计：
    - identity + background 拼成叙述段
    - traitsTags / values / taboos 拼成列表段
    - exampleDialogs 作为 few-shot 段
    - speakingStyle 作为语气指令段
    - 总长上限 ~600 token（提示词注入预算，参照 PRD 06_PRD_prompt_strategy.md）

    Args:
        persona: load_persona() 返回的字典；None 时返回空串
    """
    if not persona:
        return ""
    parts: list[str] = []

    if persona.get("identity"):
        parts.append(f"【身份】{persona['identity']}")

    if persona.get("background"):
        parts.append(f"【背景】{persona['background']}")

    tags = persona.get("traitsTags") or []
    if tags:
        parts.append("【性格标签】" + "、".join(tags))

    values = persona.get("values") or []
    if values:
        parts.append("【价值观】" + "；".join(values))

    style = persona.get("speakingStyle")
    if style:
        parts.append(f"【语气】{style}")

    taboos = persona.get("taboos") or []
    if taboos:
        parts.append("【禁忌】不要：" + "、".join(taboos))

    examples = persona.get("exampleDialogs") or []
    if examples and isinstance(examples, list):
        ex_lines = []
        for ex in examples[:3]:  # 最多 3 个示例，避免超 600 token
            if isinstance(ex, dict):
                u = ex.get("user", "")
                a = ex.get("assistant", "")
                if u and a:
                    ex_lines.append(f"用户：{u}\n你：{a}")
        if ex_lines:
            parts.append("【示例对话】\n" + "\n\n".join(ex_lines))

    return "\n\n".join(parts)


def _safe_json_list(raw: Any) -> list:
    """安全 JSON 反序列化为 list；失败/为空返回 []"""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else []
    except (json.JSONDecodeError, TypeError):
        return []