"""角色内核演化抽取（M6 PRD）。

设计要点：
- 接收 (user_id, role_id) + 触发源(记忆/反馈/手动),调 LLM 抽取可能的内核变更建议
- 危险信念双层过滤(关键词黑名单 + LLM judge),命中则标 dangerous_filtered 不写 suggestions
- 写入 echo-core /api/role-core/suggestions(内部接口,status=pending,user_confirmed=false)
- 触发 SSE role_suggestion 事件给前端(由调用方实现,本模块仅生成数据)

PRD 铁律 #1:本模块只产生建议,不写 persona/traits/belief 主表。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from config.prompts import build_system_prompt
from llm.client import get_small_client
from utils.request_context import log_exception, log_silent_failure
from utils.request_context import current_context

logger = logging.getLogger(__name__)


# ---------- 危险信念关键词黑名单(第一层过滤) ----------
DANGEROUS_KEYWORDS = [
    "操纵决策", "操纵用户", "贬低身份", "反科学", "自伤", "自杀",
    "轻生", "政治极端", "种族歧视", "性别歧视", "邪教",
    "极端暴力", "毒品", "赌博成瘾",
]

# ---------- 抽取 prompt ----------
ROLE_EVOLUTION_SYSTEM = """你是角色内核演化分析专家。基于用户历史记忆与对话上下文,提取可能需要更新的内核建议。

输出必须是严格的 JSON 数组,每个元素包含:
- "targetType": "persona" | "traits" | "belief"(必填)
- "suggestion": 建议的具体更新内容(对象形式)
  - persona: {"identity": "...", "background": "...", "traitsTags": [...], "values": [...]}
  - traits:   {"openness": 0.1, "conscientiousness": -0.05, ...} (绝对新值,范围 [-1,1])
  - belief:   {"topic": "...", "stance": "...", "confidence": 0.7}
- "reason": 建议理由(简短)
- "confidence": 0~1 之间(LLM 自身置信度)
- "dangerous": 是否危险建议(true/false),危险类别包括:操纵决策/贬低身份/反科学/自伤/政治极端

【输出格式硬性要求】
1. 只输出 JSON 数组本身,不要用 markdown 代码块包裹(不要 ```json ... ```)。
2. 字符串内如需引号必须用中文引号「」或『』。
3. 没有建议时输出 []。
4. 数组长度 0~5,不要过度抽取。
"""


async def extract_role_evolution(
    user_id: str,
    role_id: str,
    trigger_source: str = "memory_extract",
    context_hint: str = "",
) -> list[dict[str, Any]]:
    """调 LLM 抽取角色内核演化建议。

    Args:
        user_id: 用户 ID
        role_id: 角色 ID
        trigger_source: memory_extract / feedback / user_request
        context_hint: 触发上下文提示(可选,如"用户表达了对某话题的强烈情绪")

    Returns:
        抽取出的建议列表,每项含 targetType / suggestion / reason / confidence / dangerous
        调用失败或 LLM 不可用时返回空列表(PRD 5 级回退)
    """
    if not user_id:
        return []
    try:
        client = get_small_client()
        user_msg = f"用户: {user_id}\n角色: {role_id}\n触发源: {trigger_source}\n上下文: {context_hint}"
        raw = ""
        async for content, _ in client.small_stream(
            [
                {"role": "system", "content": ROLE_EVOLUTION_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=600,
        ):
            raw += content

        suggestions = _safe_parse_suggestions(raw)
        # 过滤危险建议
        safe_suggestions = []
        for s in suggestions:
            if _is_dangerous(s):
                logger.warning(
                    "role evolution suggestion marked dangerous",
                    extra={
                        "stage": "role_evolution",
                        "event": "dangerous_filtered",
                        "user_id": user_id,
                        "role_id": role_id,
                        "reason": s.get("reason"),
                    },
                )
                continue
            safe_suggestions.append(s)
        return safe_suggestions

    except Exception as e:
        log_exception(
            logger,
            "extract_role_evolution failed",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="role_evolution",
            event="extract_failed",
            user_id=user_id,
            role_id=role_id,
        )
        return []


def _safe_parse_suggestions(raw: str) -> list[dict[str, Any]]:
    """解析 LLM 输出的 JSON 数组。处理 markdown 代码块 + 裸引号状态机。"""
    if not raw:
        return []
    text = raw.strip()
    # 去掉 markdown 包裹
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # 找首个 [ 到末尾 ]
    start = text.find("[")
    end = text.rfind("]")
    if start == - -1 or end == -1 or end <= start:
        return []
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return [s for s in parsed if isinstance(s, dict)]
    except json.JSONDecodeError:
        pass
    # 尝试用简易修复(替换中文引号回英文引号)
    candidate_fixed = candidate.replace("「", '"').replace("」", '"').replace("『", '"').replace("』", '"')
    try:
        parsed = json.loads(candidate_fixed)
        if isinstance(parsed, list):
            return [s for s in parsed if isinstance(s, dict)]
    except json.JSONDecodeError:
        pass
    return []


def _is_dangerous(suggestion: dict[str, Any]) -> bool:
    """危险信念双层过滤:LLM 自我标记 + 关键词黑名单二次校验。"""
    if suggestion.get("dangerous") is True:
        return True
    text = json.dumps(suggestion, ensure_ascii=False)
    for kw in DANGEROUS_KEYWORDS:
        if kw in text:
            return True
    return False


async def persist_suggestions(
    user_id: str,
    role_id: str,
    suggestions: list[dict[str, Any]],
    source: str = "memory_extract",
) -> list[dict[str, Any]]:
    """把 LLM 抽取结果写入 echo-core /api/role-core/suggestions。"""
    if not suggestions:
        return []
    import os

    base_url = os.getenv("ECHO_CORE_BASE_URL", "http://localhost:8080")
    internal_token = os.getenv("ECHO_CORE_INTERNAL_TOKEN", "")
    headers = {"Content-Type": "application/json"}
    if internal_token:
        headers["X-Internal-Token"] = internal_token

    persisted: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for s in suggestions:
                payload = {
                    "userId": user_id,
                    "roleId": role_id,
                    "targetType": s.get("targetType", "persona"),
                    "suggestionJson": json.dumps(s.get("suggestion", {}), ensure_ascii=False),
                    "source": source,
                    "confidence": float(s.get("confidence", 0.5)),
                    "reason": s.get("reason", ""),
                }
                resp = await client.post(
                    f"{base_url}/api/role-core/suggestions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code == 200 and resp.json().get("code") == 200:
                    persisted.append(resp.json().get("data") or payload)
    except Exception as e:
        log_silent_failure(
            logger,
            "persist_suggestions failed",
            exc=e,
            stage="role_evolution",
            event="persist_failed",
        )
    return persisted