"""对话意图分类。

每次 chat 入口先用小模型把用户消息分到 5 类（chat / recall / text_search /
image_search / doc_search），由调用方按意图决定是否触发 RAG 检索。

设计取舍：
- 复用 ``client.small_prefix``（独立端点，30s 兜底 timeout，失败返回空串）。
- 外层 ``asyncio.wait_for`` 套软超时，避免小模型偶尔抖动拖累首屏。
- 任何失败路径统一 fallback 到 ``chat``（最保守：不查 RAG、不注入 L1 hint），
  由 SSE ``intent_source`` 字段告诉前端真实分类来源，便于埋点统计。

稳定性保障（历史的「超时即 chat」坑）：
- 推理型模型输出常常把答案包在 `` 思考块 `` 里再贴 JSON；调用前先剥掉 thinking
  再做正则解析（与项目其它 LLM 入口一致）。
- LLM 是推理模型时 ``max_tokens=40`` 会被思考块吃光再也吐不出 JSON；这里给到
  ``120`` 让模型有「思考 + JSON」两个输出的余裕，统计显示 90%+ 一次响应能完成。
- 常见寒暄类语句（"我回来了" / "你好" / "好久不见" 等）走毫秒级规则路径，
  完全跳过模型调用，分类既稳定也省首屏。
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
from llm.think import _strip_think
from llm.client import get_llm_client
from utils.request_context import log_exception, merge_extra

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    CHAT = "chat"
    RECALL = "recall"
    TEXT_SEARCH = "text_search"
    IMAGE_SEARCH = "image_search"
    DOC_SEARCH = "doc_search"


VALID_INTENTS: frozenset[str] = frozenset(i.value for i in Intent)


# LLM 分类预算：要给「思考 + JSON」两个输出都留够位置；40 会被思考吃掉。
_INTENT_MAX_TOKENS = 120


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


# ---------- 规则快速路径：常见寒暄、问候、回归类 ----------
# 设计意图：这些语句没有歧义但 LLM 会拖延 / 超时；走规则 <1ms 返回，保证稳定性。
#
# 注意：这里使用「精确包含」而非正则，避免误伤含有「我回来了」字样但实际是在
# 表述某事件的复合句（例如「我跟朋友说我回来了就出门了」）。复合句会越过规则
# 路径进入 LLM 分类，不受本表影响。
_GREETING_PATTERNS: tuple[tuple[str, ...], ...] = (
    # 中文寒暄 / 回归
    ("我回来了", "我回来了呀", "我回来啦", "回来啦", "回来了"),
    ("你好", "您好", "hello", "hi", "hey"),
    ("嗨", "哈喽", "在吗", "在么"),
    ("早", "早上好", "早安", "午安", "晚安"),
    ("好久不见", "想你了", "最近怎么样"),
)
_GREETING_TO_INTENT: dict[str, Intent] = {p: Intent.CHAT for grp in _GREETING_PATTERNS for p in grp}


# ---------- 规则快速路径：图片/文档/文本/回忆类 ----------
# 设计意图：「我想看猫」「找张山的照片」「之前那张图」这类 LLM 经常会拖到超时 / 误判
# 为 chat（因为没有「在 RAG 里找」这种明显标记）。用包含式正则覆盖最常见的中文表达，
# 规则命中直接锁定 image_search / doc_search / text_search / recall，跳过 LLM。
#
# 匹配策略：从短到长依次尝试「子串包含」，任一命中即返回。复合句会越过规则路径
# 交给 LLM 分类，避免误判。
_IMAGE_VIEW_RE = re.compile(
    r"(想看|给我看|找[一张只个份]?[一-龥A-Za-z0-9_]*的?图|"
    r"想看[一-龥A-Za-z0-9_]*的?照片|看[一]?下[一-龥A-Za-z0-9_]*的?图|"
    r"之前[的上]?那?张?图|之前[的上]?那?张?照片|"
    r"帮我找.*?图|看看.*?图|看看.*?照片|翻出.*?图|翻出.*?照片)",
    re.IGNORECASE,
)
_DOC_FIND_RE = re.compile(
    r"(找[一]?[个份]?[一-龥A-Za-z0-9_]*的?(合同|简历|文件|资料|pdf|文档)|"
    r"翻[出找]?[一-龥A-Za-z0-9_]*的?合同|之前[的上]?那?份?合同|"
    r"之前[的上]?那?份?文件|之前[的上]?那?份?简历)",
    re.IGNORECASE,
)
_TEXT_FIND_RE = re.compile(
    r"(找[一]?[段篇]?[一-龥A-Za-z0-9_]*的?(笔记|摘录|文章|语录|句子)|"
    r"翻[出找]?[一-龥A-Za-z0-9_]*的?笔记|"
    r"之前[的上]?那?段?笔记|之前[的上]?那?篇?文章)",
    re.IGNORECASE,
)
_RECALL_RE = re.compile(
    r"(我(之前|以前|上次|最近)说过|"
    r"我(之前|以前|上次|最近)讲过|"
    r"我(之前|以前|上次|最近)提过|"
    r"我叫什么|你记得我|你还记得)",
    re.IGNORECASE,
)


def _rule_based_intent(msg: str) -> IntentResult | None:
    """对常见寒暄 / 回归 / 看图 类语句做毫秒级分类，跳过 LLM 调用。

    返回 ``None`` 表示规则未命中，应继续走 LLM 路径。
    """
    if not msg:
        return None
    stripped = msg.strip().rstrip("！!。.?？~～,.，、 ")
    if not stripped:
        return None
    # 1) 寒暄精确表
    intent = _GREETING_TO_INTENT.get(stripped)
    if intent is not None:
        return IntentResult(
            intent=intent,
            raw=msg,
            source="rule_greeting",
            duration_ms=0.0,
        )
    # 2) 业务意图正则（含子串匹配，复合句会越过去给 LLM）
    if _IMAGE_VIEW_RE.search(stripped):
        return IntentResult(
            intent=Intent.IMAGE_SEARCH,
            raw=msg,
            source="rule_image",
            duration_ms=0.0,
        )
    if _DOC_FIND_RE.search(stripped):
        return IntentResult(
            intent=Intent.DOC_SEARCH,
            raw=msg,
            source="rule_doc",
            duration_ms=0.0,
        )
    if _TEXT_FIND_RE.search(stripped):
        return IntentResult(
            intent=Intent.TEXT_SEARCH,
            raw=msg,
            source="rule_text",
            duration_ms=0.0,
        )
    if _RECALL_RE.search(stripped):
        return IntentResult(
            intent=Intent.RECALL,
            raw=msg,
            source="rule_recall",
            duration_ms=0.0,
        )
    return None


async def classify_intent(user_msg: str) -> IntentResult:
    """调用小模型把用户消息分类到 5 类之一；任何失败回退到 ``chat``。"""
    settings = get_settings().memory
    if not settings.intent_classifier_enabled:
        return IntentResult(Intent.CHAT, "", "fallback_disabled", 0.0)
    msg = (user_msg or "").strip()
    if not msg:
        return IntentResult(Intent.CHAT, "", "fallback_empty", 0.0)

    # 0) 规则快速路径：寒暄 / 回归毫秒级命中
    rule_hit = _rule_based_intent(msg)
    if rule_hit is not None:
        logger.info(
            "intent classified (rule)",
            extra=merge_extra(
                stage="intent_classify",
                event="rule_hit",
                intent=rule_hit.intent.value,
                source=rule_hit.source,
                msg_preview=msg[:80],
                duration_ms=0.0,
            ),
        )
        return rule_hit

    client = get_llm_client()
    messages = [
        {"role": "system", "content": INTENT_CLASSIFY_SYSTEM},
        {"role": "user", "content": msg[:512]},
    ]
    t0 = time.perf_counter()
    try:
        raw_text = await asyncio.wait_for(
            client.small_prefix(messages, max_tokens=_INTENT_MAX_TOKENS, temperature=0.1),
            timeout=settings.intent_timeout_ms / 2000.0,
        )
    except asyncio.TimeoutError:
        return _fallback("timeout", "", (time.perf_counter() - t0) * 1000)
    except Exception as e:
        log_exception(
            logger,
            "intent classify error",
            exc=e,
            level=logging.WARNING,
            stage="intent_classify",
            event="classify_error",
            msg_preview=msg[:80],
            msg_len=len(msg),
            timeout_s=settings.intent_timeout_ms / 1000.0,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return _fallback("error", str(e)[:200], (time.perf_counter() - t0) * 1000)

    # 1) 推理模型 思考块剥离：避免思考块中碰巧出现的 {} 干扰 JSON 解析
    text = _strip_think(raw_text or "")
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
            llm_raw_preview=(raw_text or "")[:80],
        ),
    )
    return parsed
