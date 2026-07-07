"""对话意图分类。

每次 chat 入口先用小模型把用户消息分到 5 类（chat / recall / text_search /
image_search / doc_search），由调用方按意图决定是否触发 RAG 检索。

设计取舍：
- 复用 ``client.small_prefix``（独立端点，30s 兜底 timeout，失败返回空串）。
- 外层 ``asyncio.wait_for`` 套 1500ms 软超时，避免小模型偶尔抖动拖累首屏。
- 任何失败路径统一 fallback 到 ``chat``（最保守：不查 RAG、不注入 L1 hint），
  由 SSE ``intent_source`` 字段告诉前端真实分类来源，便于埋点统计。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum

from config.config import get_settings
from config.prompts import INTENT_CLASSIFY_SYSTEM
from llm.client import get_llm_client
from utils.request_context import merge_extra

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    CHAT = "chat"
    RECALL = "recall"
    TEXT_SEARCH = "text_search"
    IMAGE_SEARCH = "image_search"
    DOC_SEARCH = "doc_search"


VALID_INTENTS: frozenset[str] = frozenset(i.value for i in Intent)


@dataclass
class IntentResult:
    intent: Intent
    raw: str
    source: str
    duration_ms: float


def _fallback(reason: str, raw: str, duration_ms: float) -> IntentResult:
    logger.info(
        "intent fallback",
        extra=merge_extra(
            stage="intent_classify",
            event="fallback",
            reason=reason,
            raw_preview=(raw or "")[:80],
            duration_ms=round(duration_ms, 2),
        ),
    )
    return IntentResult(
        intent=Intent.CHAT,
        raw=raw or "",
        source=f"fallback_{reason}",
        duration_ms=round(duration_ms, 2),
    )


def _parse_intent_json(text: str, duration_ms: float) -> IntentResult | None:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*?\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    label = obj.get("intent") if isinstance(obj, dict) else None
    if not isinstance(label, str) or label not in VALID_INTENTS:
        return None
    return IntentResult(
        intent=Intent(label),
        raw=text,
        source="llm",
        duration_ms=round(duration_ms, 2),
    )


async def classify_intent(user_msg: str) -> IntentResult:
    """调用小模型把用户消息分类到 5 类之一；任何失败回退到 ``chat``。"""
    settings = get_settings().memory
    if not settings.intent_classifier_enabled:
        return IntentResult(Intent.CHAT, "", "fallback_disabled", 0.0)
    msg = (user_msg or "").strip()
    if not msg:
        return IntentResult(Intent.CHAT, "", "fallback_empty", 0.0)

    client = get_llm_client()
    messages = [
        {"role": "system", "content": INTENT_CLASSIFY_SYSTEM},
        {"role": "user", "content": msg[:512]},
    ]
    t0 = time.perf_counter()
    try:
        text = await asyncio.wait_for(
            client.small_prefix(messages, max_tokens=40, temperature=0.1),
            timeout=settings.intent_timeout_ms / 1000.0,
        )
    except asyncio.TimeoutError:
        return _fallback("timeout", "", (time.perf_counter() - t0) * 1000)
    except Exception as e:
        return _fallback("error", str(e)[:200], (time.perf_counter() - t0) * 1000)

    duration_ms = (time.perf_counter() - t0) * 1000
    parsed = _parse_intent_json(text, duration_ms)
    if parsed is None:
        # 解析/标签校验失败：区分两种来源以便埋点
        if not text:
            return _fallback("empty_response", "", duration_ms)
        if not re.search(r"\{[\s\S]*?\}", text):
            return _fallback("parse_error", text, duration_ms)
        return _fallback("invalid_label", text, duration_ms)

    logger.info(
        "intent classified",
        extra=merge_extra(
            stage="intent_classify",
            event="ok",
            intent=parsed.intent.value,
            msg_preview=msg[:80],
            duration_ms=parsed.duration_ms,
        ),
    )
    return parsed
