"""Whisper 音频 embedding：转录 + 声纹嵌入。

- 转录：`openai-whisper`（如未安装则返回空串与提示）。
- 声纹：使用 BGE-M3 对转录文本编码作为弱声纹特征；这样不依赖额外的声纹模型，
  且便于跨模态检索对齐。也可替换为专门的 speaker embedding（如 resemblyzer）。

设计原则：
- 模型加载失败时**绝不抛异常**，返回 `(text="", embedding=zero_vec)`，
  让上层调用方可以正常继续。
"""

from __future__ import annotations

import hashlib
import logging
import time

from config.config import get_settings
from utils.request_context import log_exception, log_silent_failure, merge_extra

logger = logging.getLogger(__name__)


def _fallback_embedding(text: str, dim: int) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals: list[float] = []
    i = 0
    while len(vals) < dim:
        vals.append((h[i % len(h)] / 255.0) * 2.0 - 1.0)
        i += 1
    return vals[:dim]


def _transcribe(audio_bytes: bytes) -> str:
    try:
        import io

        import whisper  # type: ignore

        model = getattr(whisper, "_ECHO_MODEL", None)
        if model is None:
            model = whisper.load_model("base")
            whisper._ECHO_MODEL = model  # type: ignore[attr-defined]
        # whisper 要求文件路径；落盘到临时文件后转录，再清理。
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            result = model.transcribe(tmp_path, fp16=False)
            return (result.get("text") or "").strip()
        finally:
            try:
                os.remove(tmp_path)
            except Exception as e:
                log_silent_failure(
                    logger,
                    "whisper temp file cleanup failed (skip)",
                    exc=e,
                    stage="whisper_transcribe",
                    event="tmp_cleanup_error",
                    tmp_path=tmp_path,
                )
    except Exception as e:
        log_exception(
            logger,
            "whisper transcribe failed (return empty text)",
            exc=e,
            level=logging.WARNING,
            stage="whisper_transcribe",
            event="transcribe_error",
            audio_bytes=len(audio_bytes or b""),
        )
        return ""


def embed_audio(audio_bytes: bytes) -> dict:
    """音频 → {text: str, embedding: list[float]}。失败返回空文本+零向量。"""
    cfg = get_settings()
    dim = cfg.bge_m3.dim
    t0 = time.perf_counter()
    text = _transcribe(audio_bytes) if audio_bytes else ""
    if text:
        from embedding import bge_m3

        embedding = bge_m3.embed_texts([text])[0]
    else:
        embedding = _fallback_embedding("", dim)
    logger.info(
        "whisper.embed_audio done",
        extra=merge_extra(
            stage="whisper_embed",
            event="ok",
            audio_bytes=len(audio_bytes) if audio_bytes else 0,
            text_len=len(text),
            dim=len(embedding),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        ),
    )
    return {"text": text, "embedding": embedding, "dim": dim}


def dim() -> int:
    return get_settings().bge_m3.dim