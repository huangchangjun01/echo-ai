"""音频解析器：whisper 转写（已有能力）。"""

from __future__ import annotations

import logging

from embedding import whisper
from utils.downloader import fetch_source_bytes
from utils.request_context import log_exception, merge_extra
from .base import ParsedFile, ParsedChunk

logger = logging.getLogger(__name__)


async def parse_audio(file_key: str, file_name: str, url: str | None) -> ParsedFile:
    if not file_key and not url:
        return ParsedFile(modality="audio", meta={"error": "no_key_or_url"})
    try:
        # fileKey 优先直连七牛，url 仅兜底
        raw = await fetch_source_bytes(file_key or None, url or None)
        # whisper._transcribe 内部已处理临时文件 + 清理，对外只接受 bytes
        import asyncio

        text = await asyncio.to_thread(whisper._transcribe, raw)
        if not text.strip():
            return ParsedFile(modality="audio", meta={"error": "empty_transcript"})
        logger.info(
            "audio parsed",
            extra=merge_extra(
                stage="parser_audio",
                event="ok",
                file_name=file_name,
                text_len=len(text),
            ),
        )
        return ParsedFile(
            modality="audio",
            text=text,
            chunks=[ParsedChunk(text=text, source=file_name)],
            detail_md=text,
            meta={"bytes": len(raw)},
        )
    except Exception as e:
        log_exception(
            logger,
            "parse_audio failed",
            exc=e,
            level=logging.WARNING,
            stage="parser_audio",
            event="error",
            file_name=file_name,
        )
        return ParsedFile(modality="audio", meta={"error": str(e)[:200]})