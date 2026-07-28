"""文本解析器：纯文本/代码/markdown 直接下载后按行处理。"""

from __future__ import annotations

import logging

from utils.downloader import fetch_source_bytes
from utils.request_context import log_exception
from .base import ParsedFile, ParsedChunk

logger = logging.getLogger(__name__)


async def parse_text(file_key: str, file_name: str, url: str | None) -> ParsedFile:
    if not file_key and not url:
        return ParsedFile(modality="text", text="", meta={"error": "no_key_or_url"})
    try:
        # fileKey 优先直连七牛，url 仅兜底：避免把 SPA fallback / 错误页 HTML 当正文
        raw = await fetch_source_bytes(file_key or None, url or None)
        # 简单去 BOM 与解码失败兜底
        text = raw.decode("utf-8", errors="replace")
        text = text.lstrip("﻿").strip()
        return ParsedFile(
            modality="text",
            text=text[:2000],  # 主文本截断
            chunks=[ParsedChunk(text=text, source=file_name)],
            detail_md=text,
            meta={"bytes": len(raw)},
        )
    except Exception as e:
        log_exception(
            logger,
            "parse_text failed",
            exc=e,
            level=logging.WARNING,
            stage="parser_text",
            event="error",
            file_name=file_name,
        )
        return ParsedFile(modality="text", text="", meta={"error": str(e)[:200]})