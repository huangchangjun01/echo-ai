"""7 段 system prompt 拼装(PRD 06_PRD_prompt_strategy.md)。

设计要点:
- 每段独立 token 预算(合计 <= 2300 token)
- 缺啥段就空字符串注入(不报错)
- 与原 build_system_prompt 解耦:通过 character/build_segments() 拿到 7 段 dict,
  biz/chat.py 把 dict 传给 build_system_prompt(segments=...) (保持原签名兼容)

PRD 6.1 段 system prompt 注入(硬上限 <= 2300 token):

| 段落          | 上限   | 注入时机    | 缓存策略     |
|---------------|--------|-------------|--------------|
| 人格 persona  | 600    | 每轮        | LRU by version |
| 性格底色 traits | 80   | 每 24h      | LRU by date |
| 心情 mood     | 60     | 每轮        | 不缓存       |
| 关系值        | 60     | 每轮        | 不缓存       |
| 信念摘要      | 150    | 5 分钟      | LRU          |
| L0 长期记忆   | 800    | 每轮        | 已有         |
| L1 近期摘要   | 400    | 每轮        | 已有         |
| 工具描述      | 150    | 每轮        | 已有         |
| **合计**      | **<= 2300** |         |              |

M0 阶段:本文件只搭建接口 + 7 段名 + 预算常量;具体每段的渲染逻辑在阶段 2 (M2/M4) 补全。
"""

from __future__ import annotations

import logging

from utils.request_context import log_exception

logger = logging.getLogger(__name__)


# 7 段(实际为 8 段,含 tool_descs)的 token 预算
SEGMENT_BUDGET: dict[str, int] = {
    "persona": 600,
    "traits": 80,
    "mood": 60,
    "relationship": 60,
    "belief_summary": 150,
    "L0_memories": 800,
    "L1_summ": 400,
    "tool_descs": 150,
}


async def build_segments(
    user_id: str,
    role_id: str = "default",
    *,
    l0_memories: list[str] | None = None,
    recent_summaries: list[str] | None = None,
    tool_descriptions: str | None = None,
) -> dict[str, str]:
    """拼装 7 段 system prompt 片段。

    Args:
        user_id: 用户 ID
        role_id: 角色 ID
        l0_memories: 已从 memory.retriever.load_l0_memories() 取到的 L0 列表
        recent_summaries: 已从 memory.retriever.load_l1_summaries() 取到的 L1 摘要
        tool_descriptions: 已从 config.prompts.TOOL_DESCRIPTIONS 拼好的工具描述

    Returns:
        dict 键为段名(persona/traits/mood/...),值为该段的提示词字符串(缺则空串)
        任何一段查询失败 -> 该段为空字符串(不抛错,保证主流程不中断)
    """
    segments: dict[str, str] = {k: "" for k in SEGMENT_BUDGET.keys()}

    # 1. persona 段:从 role_persona 表读(fallback to personas + DEFAULT_PERSONA)
    # 与胶囊展示共用 load_persona_segment，保证「展示 = 注入」。
    try:
        from .persona import load_persona_segment

        segments["persona"] = await load_persona_segment(user_id, role_id)
    except Exception as e:
        log_exception(
            logger,
            "build_segments.persona failed",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.inject",
            event="persona_failed",
            user_id=user_id,
        )

    # 2. traits 段:从 role_traits 表读
    try:
        from .traits import load_traits, render_traits_for_prompt

        traits = await load_traits(user_id, role_id)
        segments["traits"] = render_traits_for_prompt(traits)
    except Exception as e:
        log_exception(
            logger,
            "build_segments.traits failed",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.inject",
            event="traits_failed",
            user_id=user_id,
        )

    # 3. mood 段:M2 启用,从 role_mood 表读取三桶快照
    try:
        from .mood import load_mood, render_mood_for_prompt

        mood_snap = await load_mood(user_id, role_id)
        segments["mood"] = render_mood_for_prompt(mood_snap)
    except Exception as e:
        log_exception(
            logger,
            "build_segments.mood failed",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.inject",
            event="mood_failed",
            user_id=user_id,
        )

    # 4. relationship 段:M3 启用,从 role_relationship 表读取
    try:
        from .relationship import load_relationship, render_relationship_for_prompt

        rel = await load_relationship(user_id, role_id)
        segments["relationship"] = render_relationship_for_prompt(rel)
    except Exception as e:
        log_exception(
            logger,
            "build_segments.relationship failed",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.inject",
            event="relationship_failed",
            user_id=user_id,
        )

    # 5. belief_summary 段:M3 启用,从 role_belief 表读取已确认信念(5 分钟 LRU)
    try:
        from .belief_summary import load_belief_summary, render_belief_summary_for_prompt

        belief_text = await load_belief_summary(user_id, role_id)
        segments["belief_summary"] = render_belief_summary_for_prompt(belief_text)
    except Exception as e:
        log_exception(
            logger,
            "build_segments.belief_summary failed",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.inject",
            event="belief_summary_failed",
            user_id=user_id,
        )

    # 6. L0_memories 段:直接复用 biz/chat.py 已查到的列表
    if l0_memories:
        segments["L0_memories"] = "\n".join(f"- {m}" for m in l0_memories[:20])

    # 7. L1_summ 段:直接复用
    if recent_summaries:
        segments["L1_summ"] = "\n".join(f"- {m}" for m in recent_summaries[:10])

    # 8. tool_descs 段
    if tool_descriptions:
        segments["tool_descs"] = tool_descriptions

    # 预算裁剪(粗略按字符数 1 token ~= 1.5 字符估算)
    for name, budget in SEGMENT_BUDGET.items():
        max_chars = int(budget * 1.5)
        if len(segments[name]) > max_chars:
            segments[name] = segments[name][:max_chars] + "..."  # noqa: E501

    total_chars = sum(len(s) for s in segments.values())
    logger.debug(
        "build_segments ok",
        extra={
            "stage": "character.inject",
            "event": "ok",
            "user_id": user_id,
            "role_id": role_id,
            "total_chars": total_chars,
            "non_empty": [k for k, v in segments.items() if v],
        },
    )
    return segments