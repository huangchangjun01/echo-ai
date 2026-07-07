"""流式级联：小模型快速前缀 + 大模型深度续写。

设计目标：用户感知到的首字延迟尽可能短（< 200ms），同时拥有大模型的深度续写质量。

实现策略（实际可行方案）：
- 小模型：用于快速产出"前缀"，让用户在第一时间看到回答的开头。
- 大模型：完整生成完整回复（含前缀内容）。我们尝试在重叠前缀的位置之后才开始 yield delta。
- 任何无法去重的情况（如两个模型输出完全不同的开场），cascade 选择保留小模型输出 + 大模型整体输出（用户能看到两段）。

用户可见的最终文本 = max overlap removed 大模型输出 + 之前的小模型前缀（如果未被大模型覆盖）。
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


def _find_overlap(prefix: str, full: str, min_chars: int = 8) -> int:
    """返回 `full` 中与 `prefix` 末尾重叠的最长前缀长度（字符级）。

    启发式：
    1. 整段匹配：prefix 是否作为 substring 出现在 full 中 → 命中后 split 即可。
    2. 后缀匹配：prefix 的后 K 字符是否等于 full 的前 K 字符（K>=min_chars）。
    """
    if not prefix or not full:
        return 0
    p = prefix.strip()
    if p and p in full:
        # prefix 完整出现在 full 中：返回 prefix 长度
        return len(p)
    # 字符级最大后缀前缀匹配
    limit = min(len(p), len(full))
    for k in range(limit, min_chars - 1, -1):
        if p[-k:] == full[:k]:
            return k
    return 0


async def cascade_chat(
    messages: list[dict[str, str]],
    *,
    prefix_tokens: int = 64,
) -> AsyncIterator[dict[str, Any]]:
    """流式级联生成。

    yield 事件类型：
    - {"type": "prefix", "text": str}
    - {"type": "delta", "text": str}
    - {"type": "done", "full": str}
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

    # Step 1: 小模型快速前缀（给用户首屏）
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

    # Step 2: 大模型完整生成（流式）
    cascaded_messages = list(messages)
    if prefix:
        cascaded_messages = cascaded_messages + [{"role": "assistant", "content": prefix}]

    # 我们让大模型自由生成完整回复（不去强制它从 prefix 续写）。
    # 收集完整大模型输出后做 overlap 检测，去掉 prefix 部分。
    t_big = time.perf_counter()
    full_chunks: list[str] = []
    async for delta in client.stream(cascaded_messages):
        full_chunks.append(delta)
    big_full = "".join(full_chunks)
    big_clean = _strip_think(big_full)
    big_ms = round((time.perf_counter() - t_big) * 1000, 2)

    # 计算 overlap
    overlap = _find_overlap(prefix, big_clean, min_chars=4)
    if overlap > 0 and len(big_clean) > overlap:
        tail = big_clean[overlap:]
        final_text = prefix + tail
        overlap_kind = "suffix_prefix_match"
    elif overlap == 0 and big_clean.startswith(prefix):
        tail = big_clean[len(prefix):]
        final_text = prefix + tail
        overlap_kind = "startswith"
    else:
        # 无可识别重叠：以大模型完整输出为准（更权威）。
        tail = big_clean
        final_text = big_clean
        prefix = ""  # 前缀被覆盖，避免被外层拼接
        overlap_kind = "none"

    logger.info(
        "cascade overlap",
        extra=merge_extra(
            stage="cascade",
            event="overlap",
            overlap_chars=overlap if overlap_kind != "startswith" else len(prefix),
            overlap_kind=overlap_kind,
            big_full_len=len(big_full),
            big_clean_len=len(big_clean),
            tail_len=len(tail),
            prefix_kept=bool(prefix),
            big_model_ms=big_ms,
        ),
    )

    # Step 3: 增量 yield tail（保证流式体验）
    chunk_size = 8
    emitted = 0
    delta_count = 0
    while emitted < len(tail):
        piece = tail[emitted : emitted + chunk_size]
        emitted += len(piece)
        delta_count += 1
        yield {"type": "delta", "text": piece}

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