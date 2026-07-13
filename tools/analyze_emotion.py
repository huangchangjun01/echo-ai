"""analyze_emotion：进程内情感分析。

- 优先使用 LLM 做细粒度情感分析（情感标签 + 强度）。
- 失败时退化为简易关键词规则，保证不抛异常。
- 结果同时持久化到 emotion_logs（异步）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from tools.base import BaseTool, ToolResult, ok
from utils.request_context import log_exception

logger = logging.getLogger(__name__)

_EMOTION_RULES = {
    "joy": ("开心", "高兴", "快乐", "喜欢", "感谢", "happy", "glad", "love"),
    "sadness": ("难过", "伤心", "失落", "孤独", "sad", "lonely"),
    "anger": ("生气", "愤怒", "烦", "讨厌", "angry"),
    "surprise": ("惊讶", "意外", "震惊", "wow"),
    "fear": ("担心", "害怕", "焦虑", "afraid"),
}


def _rule_based(text: str) -> dict:
    t = (text or "").lower()
    scores = {k: 0.0 for k in _EMOTION_RULES}
    for emo, kws in _EMOTION_RULES.items():
        for kw in kws:
            if kw in t:
                scores[emo] += 1.0
    if max(scores.values()) == 0:
        return {"emotion": "neutral", "intensity": 0.0, "reason": "no signal"}
    emo = max(scores, key=lambda x: scores[x])
    intensity = min(1.0, scores[emo] / 3.0)
    return {"emotion": emo, "intensity": round(intensity, 3), "reason": "rule-based"}


class AnalyzeEmotionTool(BaseTool):
    name = "analyze_emotion"
    description = (
        "analyze_emotion(text: str, user_id: str = '') -> 对文本做细粒度情感分析。"
        "返回 {emotion: str, intensity: float, reason: str}。"
        "当用户表达复杂情绪或需要回应情绪时使用。"
    )

    async def arun(self, text: str = "", user_id: str = "", **_) -> ToolResult:
        return await _analyze_emotion_async(text=text, user_id=user_id)

    def run(self, text: str = "", user_id: str = "", **_) -> ToolResult:  # noqa: D401
        try:
            asyncio.get_running_loop()
            from tools.base import fail

            return fail("Use arun() in async context")
        except RuntimeError:
            return asyncio.run(_analyze_emotion_async(text=text, user_id=user_id))


async def _analyze_emotion_async(*, text: str, user_id: str) -> ToolResult:
    if not text:
        return ok({"emotion": "neutral", "intensity": 0.0, "reason": "empty"})

    # 优先 LLM
    try:
        from llm.client import get_llm_client
        from config.prompts import EMOTION_ANALYZE_SYSTEM

        client = get_llm_client()
        resp = await client.chat(
            [
                {"role": "system", "content": EMOTION_ANALYZE_SYSTEM},
                {"role": "user", "content": text},
            ],
            max_tokens=120,
            temperature=0.2,
        )
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        m = re.search(r"\{[\s\S]*\}", content or "")
        if m:
            obj = json.loads(m.group(0))
            emo = obj.get("emotion", "neutral")
            intensity = float(obj.get("intensity", 0.0) or 0.0)
            reason = obj.get("reason", "")
            await _persist_emotion(user_id, text, emo, intensity, reason)
            return ok({"emotion": emo, "intensity": intensity, "reason": reason})
    except Exception as e:
        log_exception(
            logger,
            "LLM emotion failed, fallback rule",
            exc=e,
            level=logging.WARNING,
            stage="analyze_emotion",
            event="llm_error",
            user_id=user_id,
            text_preview=(text or "")[:80],
            text_len=len(text or ""),
        )

    # 回退：规则
    obj = _rule_based(text)
    await _persist_emotion(user_id, text, obj["emotion"], obj["intensity"], obj["reason"])
    return ok(obj)


async def _persist_emotion(user_id: str, text: str, emotion: str, intensity: float, reason: str) -> None:
    if not user_id:
        return
    try:
        from database import execute

        await execute(
            """
            INSERT INTO emotion_logs (user_id, text, emotion, intensity, reason)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_id, text[:1000], emotion, float(intensity), (reason or "")[:255]),
        )
    except Exception as e:
        log_exception(
            logger,
            "persist emotion log failed",
            exc=e,
            level=logging.WARNING,
            stage="analyze_emotion",
            event="persist_error",
            user_id=user_id,
            emotion=emotion,
            intensity=intensity,
        )