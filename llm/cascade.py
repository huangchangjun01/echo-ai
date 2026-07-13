"""流式级联：小模型快速前缀 + 大模型实时续写。

设计目标：用户感知到的首字延迟尽可能短，同时拥有大模型的深度续写质量。

实现策略：
- 小模型：快速产出"前缀"，先 yield 给用户，占住首屏。
- 大模型：以「续写」语义接着前缀往下写，**逐 chunk 实时 yield**（不再先缓冲整段再切片）。
- 前缀一旦 yield 出去就不可撤回，因此最终文本恒等于 ``prefix + 大模型续写尾``，
  绝不丢弃已展示的前缀。大模型若无视指令把开头又抄了一遍，用门控去重跳过重复段。
- ``<think>...</think>`` 推理块在续写流里增量剥离：think 未闭合时按住不吐，闭合后再吐后续。
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from llm.client import get_llm_client
from utils.request_context import merge_extra

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.MULTILINE)


def _strip_think(text: str) -> str:
    """剥离推理型模型输出的 <think>...</think> 块。"""
    if not text:
        return text
    return _THINK_RE.sub("", text).strip()


def _has_open_think(text: str) -> bool:
    """是否存在尚未闭合的 <think>（此时 cleaned 里会残留原文，需按住不吐）。"""
    return text.count("<think>") > text.count("</think>")


def _common_prefix_len(a: str, b: str) -> int:
    """a、b 从头逐字符相同的长度。"""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# 续写提示：告诉大模型"回复已经以 prefix 开头，请无缝续写、不要重复开头"。
_CONTINUE_HINT = (
    "你的回复已经以下面这段开头，请**直接接着往下写完整回复**，"
    "不要重复这段开头、不要另起炉灶、不要加引号或解释：\n「{prefix}」"
)


async def cascade_chat(
    messages: list[dict[str, str]],
    *,
    prefix_tokens: int = 64,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """流式级联生成：小模型前缀 + 大模型实时续写。

    yield 事件类型：
    - {"type": "prefix", "text": str}   小模型前缀（首屏）
    - {"type": "delta", "text": str}    大模型续写增量（实时）
    - {"type": "done", "full": str}     最终完整文本 = prefix + 续写尾
    """
    client = get_llm_client()
    t0 = time.perf_counter()
    logger.info(
        "cascade start",
        extra=merge_extra(
            stage="cascade",
            event="start",
            msg_count=len(messages),
            prefix_tokens=prefix_tokens,
        ),
    )

    # Step 1: 小模型快速前缀（首屏）
    t_prefix = time.perf_counter()
    prefix_raw = await client.small_prefix(messages, max_tokens=prefix_tokens)
    prefix = _strip_think(prefix_raw)
    prefix_ms = round((time.perf_counter() - t_prefix) * 1000, 2)
    yield {"type": "prefix", "text": prefix}
    logger.info(
        "cascade prefix ready",
        extra=merge_extra(
            stage="cascade",
            event="prefix",
            prefix_len=len(prefix),
            duration_ms=prefix_ms,
        ),
    )

    # Step 2: 大模型以「续写」语义接着 prefix 实时流式写下去
    cascaded = list(messages)
    if prefix:
        cascaded = cascaded + [
            {"role": "assistant", "content": prefix},
            {"role": "system", "content": _CONTINUE_HINT.format(prefix=prefix)},
        ]

    t_big = time.perf_counter()
    raw = ""            # 大模型原始累积（可能含 <think>）
    skip: int | None = None   # cleaned 开头需跳过的重复前缀字符数（None=未定）
    emitted = 0         # 已 yield 的 cleaned[skip:] 字符数
    tail_parts: list[str] = []
    delta_count = 0
    first_delta_ms: float | None = None

    async for chunk in client.stream(cascaded, temperature=temperature, max_tokens=max_tokens):
        raw += chunk
        if _has_open_think(raw):
            continue
        cleaned = _strip_think(raw)
        if not cleaned:
            continue

        # 门控去重：判断大模型是否把 prefix 开头又抄了一遍
        if skip is None:
            if prefix:
                common = _common_prefix_len(prefix, cleaned)
                # cleaned 目前与 prefix 完全一致且尚未超过 prefix 长度 → 可能仍在重抄，继续缓冲
                if common == len(cleaned) and len(cleaned) < len(prefix):
                    continue
                skip = len(prefix) if cleaned.startswith(prefix) else 0
            else:
                skip = 0

        usable = cleaned[skip:]
        if len(usable) > emitted:
            piece = usable[emitted:]
            emitted = len(usable)
            tail_parts.append(piece)
            delta_count += 1
            if first_delta_ms is None:
                first_delta_ms = round((time.perf_counter() - t_big) * 1000, 2)
            yield {"type": "delta", "text": piece}

    # 流结束时若仍未定 skip（大模型只回了 prefix 或空），补一次决断 + flush
    if skip is None:
        cleaned = _strip_think(raw)
        skip = len(prefix) if (prefix and cleaned.startswith(prefix)) else 0
        usable = cleaned[skip:]
        if len(usable) > emitted:
            piece = usable[emitted:]
            emitted = len(usable)
            tail_parts.append(piece)
            delta_count += 1
            yield {"type": "delta", "text": piece}

    final_text = prefix + "".join(tail_parts)
    big_ms = round((time.perf_counter() - t_big) * 1000, 2)
    logger.info(
        "cascade big model streamed",
        extra=merge_extra(
            stage="cascade",
            event="stream",
            prefix_kept=bool(prefix),
            skip_chars=skip or 0,
            raw_len=len(raw),
            tail_len=len("".join(tail_parts)),
            delta_count=delta_count,
            first_delta_ms=first_delta_ms,
            big_model_ms=big_ms,
        ),
    )

    yield {"type": "done", "full": final_text}
    logger.info(
        "cascade end",
        extra=merge_extra(
            stage="cascade",
            event="end",
            final_len=len(final_text),
            delta_count=delta_count,
            total_ms=round((time.perf_counter() - t0) * 1000, 2),
        ),
    )


async def cascade_collect(messages: list[dict[str, str]], *, prefix_tokens: int = 64) -> dict:
    """一次性收集级联结果。"""
    chunks: list[str] = []
    prefix = ""
    full = ""
    async for ev in cascade_chat(messages, prefix_tokens=prefix_tokens):
        if ev["type"] == "prefix":
            prefix = ev["text"]
        elif ev["type"] == "delta":
            chunks.append(ev["text"])
        elif ev["type"] == "done":
            full = ev["full"]
    return {
        "prefix": prefix,
        "tail": "".join(chunks),
        "full": full,
    }
