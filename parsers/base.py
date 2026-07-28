"""解析器统一接口与基础结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedChunk:
    """单条解析片段：含文本与来源标识（多模态解析时会标注来源文件）。"""
    text: str
    source: str  # 文件名（或 'description' 兜底）


@dataclass
class ParsedFile:
    """单文件解析结果。"""
    modality: str = "text"  # text/image/audio/video
    text: str = ""           # 合并后的主文本（用于摘要/总览）
    chunks: list[ParsedChunk] = field(default_factory=list)
    detail_md: str = ""      # 该文件贡献的细节段（进入 md 的 `## 记忆细节`）
    meta: dict[str, Any] = field(default_factory=dict)