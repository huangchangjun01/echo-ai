"""推理块（<think>...</think>）增量剥离工具。

供对话 / 意图 / 记忆抽取等多个 LLM 调用点复用：
- ``_strip_think``      剥离完整 <think> 块（确定性正则替换）。
- ``_has_open_think``   判断流式累积文本里是否存在未闭合的 <think>，
                        用于流式场景"未闭合时按住不吐，闭合后再吐后续"。
"""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.MULTILINE)


def _strip_think(text: str) -> str:
    """剥离推理型模型输出的 <think>...</think> 块。"""
    if not text:
        return text
    return _THINK_RE.sub("", text).strip()


def _has_open_think(text: str) -> bool:
    """是否存在尚未闭合的 <think>（此时 cleaned 里会残留原文，需按住不吐）。"""
    return text.count("<think>") > text.count("</think>")
