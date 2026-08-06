"""音频解析器：whisper 转写（已有能力）。"""

from __future__ import annotations

import logging

from embedding import whisper
from utils.downloader import fetch_source_bytes
from utils.request_context import log_exception, merge_extra
from .base import ParsedFile, ParsedChunk

logger = logging.getLogger(__name__)

# 转写产物同样需要防御性上限：一段 1 小时的录音 whisper 输出的 transcript
# 可达数 KB，与 build_memory_md 拼起来容易撞 MiniMax-M3 的上下文窗口。
_MAX_AUDIO_DETAIL_CHARS = 4000


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
        # 防御性截断（与视频解析同源修复：见 parsers/video_parser.py 注释）。
        truncated = len(text) > _MAX_AUDIO_DETAIL_CHARS
        capped = text[:_MAX_AUDIO_DETAIL_CHARS] if truncated else text
        logger.info(
            "audio parsed",
            extra=merge_extra(
                stage="parser_audio",
                event="ok",
                file_name=file_name,
                text_len=len(text),
                truncated=truncated,
            ),
        )
        return ParsedFile(
            modality="audio",
            text=text[:3000],  # 摘要向量用更短的 text 字段
            chunks=[ParsedChunk(text=capped, source=file_name)],
            detail_md=capped,
            meta={"bytes": len(raw), "truncated": truncated},
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