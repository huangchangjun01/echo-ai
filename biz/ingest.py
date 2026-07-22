from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.config import get_settings
from embedding.embeddings import ChineseCLIPEmbeddings
from embedding.models import compute_image_embeddings, compute_text_embeddings
from utils.downloader import DownloadError, download_file_async
from utils.request_context import log_exception, log_silent_failure, log_stage, merge_extra

logger = logging.getLogger(__name__)

# Mapping of MIME prefixes to logical content kinds. Detection order matters:
# more specific kinds (image) are checked before generic text.
SUPPORTED_IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif", "image/bmp"}
SUPPORTED_TEXT_MIMES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/x-python",
    "application/json",
    "application/x-ndjson",
    "text/x-log",
}
# 音频 / 视频：解析后生成记忆（音频转写、视频关键帧描述）。
SUPPORTED_AUDIO_MIMES = {"audio/mpeg", "audio/ogg", "audio/flac", "audio/wav", "audio/mp4"}
SUPPORTED_VIDEO_MIMES = {"video/mp4", "video/webm", "video/quicktime"}
# 仍明确拒绝的二进制容器（压缩包 / 未知二进制）。
REJECTED_BINARY_MIMES = {
    "application/zip",
    "application/octet-stream",
    "application/x-binary",
}


@dataclass
class IngestResult:
    success: bool
    file_id: str
    chunks: int = 0
    error: str | None = None
    parsed_text: str = ""  # 解析出的文本内容（喂给记忆抽取）
    modality: str = "text"


def _detect_mime(data: bytes) -> str:
    """Detect MIME via libmagic when available; fall back to a tiny magic-byte sniffer."""
    try:
        import magic  # type: ignore

        return magic.from_buffer(data, mime=True) or "application/octet-stream"
    except Exception as e:
        log_silent_failure(
            logger,
            "libmagic unavailable, use builtin sniffer",
            exc=e,
            stage="ingest_mime",
            event="magic_import_error",
        )
        if not data:
            return "application/octet-stream"
        head = data[:16]
        if head.startswith(b"\x89PNG"):
            return "image/png"
        if head[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if head[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            return "image/webp"
        # Reject known binary containers instead of guessing text.
        if head[4:8] == b"ftyp":  # MP4 / MOV / M4A
            return "video/mp4"
        if head[:3] == b"ID3" or (head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):  # MP3
            return "audio/mpeg"
        if head[:4] == b"OggS":  # OGG
            return "audio/ogg"
        if head[:4] == b"fLaC":  # FLAC
            return "audio/flac"
        if head[:4] == b"\x1aE\xdf\xa3":  # Matroska / WebM
            return "video/webm"
        if head[:2] in (b"PK", b"\x1f\x8b", b"\x42\x5a"):  # ZIP, gzip, bzip2
            return "application/zip"
        # Heuristic: if the first ~512 bytes mostly decode as printable utf-8/gbk, treat as text.
        sample = data[:512]
        for enc in ("utf-8", "gbk"):
            try:
                sample.decode(enc)
                return "text/plain"
            except Exception as e:
                # Allow lone invalid tail bytes (common in CJK when the cut falls mid-codepoint).
                log_silent_failure(
                    logger,
                    "strict sample decode failed; evaluating lenient ratio",
                    exc=e,
                    stage="ingest_mime",
                    event="sample_decode_strict_error",
                    encoding=enc,
                )
                bad = sample.decode(enc, errors="ignore").count("�")
                if bad <= max(1, len(sample) // 64):
                    return "text/plain"
        return "application/octet-stream"


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return data.decode(enc)
        except Exception as e:
            log_silent_failure(
                logger,
                "decode_text: enc failed, try next",
                exc=e,
                stage="ingest_decode",
                event="enc_decode_error",
                encoding=enc,
            )
            continue
    return data.decode("utf-8", errors="replace")


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []
    settings = get_settings().embedding
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )
        return [c for c in splitter.split_text(text) if c.strip()]
    except Exception as e:
        log_silent_failure(
            logger,
            "langchain splitter unavailable, use fixed-size window",
            exc=e,
            stage="ingest_split",
            event="splitter_load_error",
            text_len=len(text or ""),
        )
        # Fallback: fixed-size windows.
        step = max(1, chunk_size - chunk_overlap)
        return [text[i : i + chunk_size] for i in range(0, max(1, len(text)), step)]


async def _download_with_retry(url: str) -> bytes:
    settings = get_settings().ingest
    last_exc: Exception | None = None
    attempt_idx = 0
    t0 = time.perf_counter()
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max(1, settings.download_retries)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((DownloadError, asyncio.TimeoutError)),
        reraise=True,
    ):
        with attempt:
            attempt_idx += 1
            try:
                data = await download_file_async(url)
                logger.info(
                    "download ok",
                    extra=merge_extra(
                        stage="download",
                        event="ok",
                        attempt=attempt_idx,
                        bytes=len(data),
                        url=url,
                        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    ),
                )
                return data
            except Exception as e:
                last_exc = e
                log_exception(
                    logger,
                    "download attempt failed",
                    exc=e,
                    level=logging.WARNING,
                    stage="download",
                    event="retry",
                    attempt=attempt_idx,
                    max_attempts=max(1, settings.download_retries),
                    url=url,
                    timeout_s=settings.download_timeout_seconds,
                    elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
                raise
    if last_exc:
        raise last_exc
    raise DownloadError("download failed without exception")


def _resolve_url(file_key: str | None, url: str | None) -> str:
    if url:
        return url
    if file_key and get_settings().qiniu.base_url:
        return get_settings().qiniu.base_url.rstrip("/") + "/" + file_key.lstrip("/")
    return ""


async def ingest_file(
    user_id: str,
    file_obj: dict[str, Any],
    embeddings: ChineseCLIPEmbeddings,
    vectorstore: Any,
    desc: str | None = None,
    role_id: str = "default",
) -> IngestResult:
    """Download, classify, embed, persist a single file, then derive role-scoped memories.

    `desc` 为用户为文件填写的文字描述；`role_id` 为角色隔离标识。
    当既无 url 也无 fileKey 但有 desc 时，走「纯文本记忆」分支（不下载，直接把 desc 入库）。
    """
    file_id = file_obj.get("fileId") or ""
    file_name = file_obj.get("fileName") or ""
    file_key = file_obj.get("fileKey")
    url = _resolve_url(file_key, file_obj.get("url"))
    role_id = role_id or "default"
    desc = (desc or "").strip()
    t0 = time.perf_counter()
    logger.info(
        "ingest start",
        extra=merge_extra(
            stage="ingest",
            event="start",
            file_id=file_id,
            file_name=file_name,
            role_id=role_id,
            has_file_key=bool(file_key),
            has_desc=bool(desc),
            url=url,
        ),
    )

    # 纯文本记忆分支：无可下载资源但有描述 → 把 desc 作为文本片段入 EchoDoc + 生成记忆。
    if not url:
        if desc:
            result = await _ingest_text(
                user_id, file_id, file_name, "", desc, embeddings, vectorstore, role_id=role_id
            )
            await _maybe_generate_memory(user_id, role_id, file_id, file_name, desc, result)
            result_log_common(result, t0)
            return result
        logger.error(
            "ingest missing url",
            extra=merge_extra(
                stage="ingest",
                event="error",
                file_id=file_id,
                file_name=file_name,
                has_file_key=bool(file_key),
                error="missing url",
            ),
        )
        return IngestResult(False, file_id, error="Missing URL")

    try:
        data = await _download_with_retry(url)
    except Exception as e:
        log_exception(
            logger,
            "ingest download failed",
            exc=e,
            stage="ingest",
            event="download_error",
            file_id=file_id,
            url=url,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return IngestResult(False, file_id, error=f"Download failed: {e}")

    mime = _detect_mime(data)
    logger.info(
        "ingest mime detected",
        extra=merge_extra(
            stage="ingest",
            event="mime",
            file_id=file_id,
            mime=mime,
            bytes=len(data),
        ),
    )

    if mime in SUPPORTED_IMAGE_MIMES:
        result = await _ingest_image(
            user_id, file_id, file_name, url, data, embeddings, vectorstore, role_id=role_id
        )
    elif mime in SUPPORTED_AUDIO_MIMES:
        result = await _ingest_audio(
            user_id, file_id, file_name, url, data, vectorstore, role_id=role_id
        )
    elif mime in SUPPORTED_VIDEO_MIMES:
        result = await _ingest_video(
            user_id, file_id, file_name, url, data, vectorstore, role_id=role_id
        )
    elif mime.startswith("text/") or mime in SUPPORTED_TEXT_MIMES:
        text = _decode_text(data)
        logger.info(
            "ingest text decoded",
            extra=merge_extra(
                stage="ingest",
                event="text_decoded",
                file_id=file_id,
                text_len=len(text),
                mime=mime,
            ),
        )
        result = await _ingest_text(
            user_id, file_id, file_name, url, text, embeddings, vectorstore, role_id=role_id
        )
    elif mime in REJECTED_BINARY_MIMES:
        logger.error(
            "ingest rejected mime",
            extra=merge_extra(
                stage="ingest",
                event="rejected",
                file_id=file_id,
                file_name=file_name,
                url=url,
                mime=mime,
                bytes=len(data),
            ),
        )
        return IngestResult(False, file_id, error=f"Unsupported binary content type: {mime}")
    else:
        logger.error(
            "ingest unsupported mime",
            extra=merge_extra(
                stage="ingest",
                event="unsupported_mime",
                file_id=file_id,
                file_name=file_name,
                url=url,
                mime=mime,
                bytes=len(data),
            ),
        )
        return IngestResult(False, file_id, error=f"Unsupported content type: {mime}")

    await _maybe_generate_memory(user_id, role_id, file_id, file_name, desc, result)
    result_log_common(result, t0)
    return result


async def _maybe_generate_memory(
    user_id: str,
    role_id: str,
    file_id: str,
    file_name: str,
    desc: str,
    result: IngestResult,
) -> None:
    """入库成功后，把 desc + 解析内容交给记忆抽取（best-effort，失败不影响入库）。"""
    if not result.success:
        return
    parsed = (result.parsed_text or "").strip()
    if not desc and not parsed:
        return
    try:
        from memory import extract_from_file

        await extract_from_file(
            user_id=user_id,
            role_id=role_id,
            file_name=file_name,
            modality=result.modality,
            desc=desc,
            parsed_content=parsed,
            source_meta={"fileId": file_id},
        )
    except Exception as e:
        log_exception(
            logger,
            "file memory generation failed (non-fatal)",
            exc=e,
            level=logging.WARNING,
            stage="ingest_memory",
            event="error",
            file_id=file_id,
            role_id=role_id,
            modality=result.modality,
        )


def result_log_common(result: IngestResult, t0: float) -> None:
    """统一记录 ingest 终态日志。"""
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    if result.success:
        logger.info(
            "ingest ok",
            extra=merge_extra(
                stage="ingest",
                event="ok",
                file_id=result.file_id,
                chunks=result.chunks,
                duration_ms=elapsed,
            ),
        )
    else:
        logger.error(
            "ingest failed",
            extra=merge_extra(
                stage="ingest",
                event="error",
                file_id=result.file_id,
                err=result.error or "",
                chunks=result.chunks,
                duration_ms=elapsed,
            ),
        )


async def _ingest_text(
    user_id: str,
    file_id: str,
    file_name: str,
    url: str,
    text: str,
    embeddings: ChineseCLIPEmbeddings,
    vectorstore: Any,
    role_id: str = "default",
) -> IngestResult:
    ingest_settings = get_settings().ingest
    embed_settings = get_settings().embedding
    if not ingest_settings.enable_chunking or len(text) <= ingest_settings.max_download_bytes // 1024:
        chunks = [text] if text else []
        chunk_strategy = "passthrough"
    else:
        chunks = _split_text(text, chunk_size=0, chunk_overlap=0)
        chunk_strategy = "split"

    logger.info(
        "ingest chunks ready",
        extra=merge_extra(
            stage="ingest_text",
            event="chunks",
            file_id=file_id,
            chunk_count=len(chunks),
            chunk_strategy=chunk_strategy,
            chunk_size=embed_settings.chunk_size,
            chunk_overlap=embed_settings.chunk_overlap,
        ),
    )

    if not chunks:
        return IngestResult(False, file_id, error="Empty text content")

    try:
        t = time.perf_counter()
        vectors = await asyncio.to_thread(compute_text_embeddings, chunks)
        logger.info(
            "text embedding ok",
            extra=merge_extra(
                stage="ingest_text",
                event="embed_ok",
                file_id=file_id,
                vector_count=len(vectors),
                dim=len(vectors[0]) if vectors else 0,
                duration_ms=round((time.perf_counter() - t) * 1000, 2),
            ),
        )
    except Exception as e:
        log_exception(
            logger,
            "text embedding failed",
            exc=e,
            stage="ingest_text",
            event="embed_error",
            file_id=file_id,
            url=url,
            chunk_count=len(chunks),
            chunk_strategy=chunk_strategy,
            embed_model=embed_settings.model_name,
        )
        return IngestResult(False, file_id, error=f"Embedding failed: {e}")

    base_meta = {
        "fileId": file_id,
        "fileName": file_name,
        "userId": user_id,
        "roleId": role_id,
        "sourceUrl": url,
        "totalChunks": len(chunks),
        "modality": "text",
    }
    ids = [f"{file_id}:{i}" for i in range(len(chunks))]
    metadatas = [{**base_meta, "chunkIndex": i} for i in range(len(chunks))]
    try:
        t = time.perf_counter()
        vectorstore.add_texts(ids=ids, texts=chunks, metadatas=metadatas, embeddings=vectors)
        logger.info(
            "vector write ok",
            extra=merge_extra(
                stage="ingest_text",
                event="vector_write_ok",
                file_id=file_id,
                count=len(ids),
                duration_ms=round((time.perf_counter() - t) * 1000, 2),
            ),
        )
    except Exception as e:
        log_exception(
            logger,
            "vector store write failed",
            exc=e,
            stage="ingest_text",
            event="vector_write_error",
            file_id=file_id,
            chunk_count=len(chunks),
            url=url,
        )
        return IngestResult(False, file_id, error=f"Vector store failed: {e}")
    return IngestResult(True, file_id, chunks=len(chunks), parsed_text=text, modality="text")


async def _ingest_image(
    user_id: str,
    file_id: str,
    file_name: str,
    url: str,
    data: bytes,
    embeddings: ChineseCLIPEmbeddings,
    vectorstore: Any,
    role_id: str = "default",
) -> IngestResult:
    try:
        t = time.perf_counter()
        vectors = await asyncio.to_thread(compute_image_embeddings, [data])
        logger.info(
            "image embedding ok",
            extra=merge_extra(
                stage="ingest_image",
                event="embed_ok",
                file_id=file_id,
                dim=len(vectors[0]) if vectors else 0,
                duration_ms=round((time.perf_counter() - t) * 1000, 2),
            ),
        )
    except Exception as e:
        log_exception(
            logger,
            "image embedding failed",
            exc=e,
            stage="ingest_image",
            event="embed_error",
            file_id=file_id,
            url=url,
            image_bytes=len(data or b""),
        )
        return IngestResult(False, file_id, error=f"Image embedding failed: {e}")

    metadata = {
        "fileId": file_id,
        "fileName": file_name,
        "userId": user_id,
        "roleId": role_id,
        "sourceUrl": url,
        "chunkIndex": 0,
        "totalChunks": 1,
        "modality": "image",
    }
    try:
        t = time.perf_counter()
        vectorstore.add_texts(ids=[file_id], texts=[file_name or ""], metadatas=[metadata], embeddings=vectors)
        logger.info(
            "vector write ok",
            extra=merge_extra(
                stage="ingest_image",
                event="vector_write_ok",
                file_id=file_id,
                count=1,
                duration_ms=round((time.perf_counter() - t) * 1000, 2),
            ),
        )
    except Exception as e:
        log_exception(
            logger,
            "vector store write failed",
            exc=e,
            stage="ingest_image",
            event="vector_write_error",
            file_id=file_id,
            url=url,
        )
        return IngestResult(False, file_id, error=f"Vector store failed: {e}")

    # 图像描述（best-effort）：交给 understand_image（CLIP+LLM）产出文字，供记忆抽取使用。
    description = await _describe_image(url, file_id)
    return IngestResult(True, file_id, chunks=1, parsed_text=description, modality="image")


async def _describe_image(url: str, file_id: str) -> str:
    """调用 understand_image 生成图片文字描述；失败返回空串。"""
    if not url:
        return ""
    try:
        from tools.understand_image import _understand_image_async

        result = await _understand_image_async(url)
        if result.ok:
            return (result.data or {}).get("description", "") or ""
    except Exception as e:
        log_exception(
            logger,
            "image description failed (non-fatal)",
            exc=e,
            level=logging.WARNING,
            stage="ingest_image",
            event="describe_error",
            file_id=file_id,
            url=url,
        )
    return ""


async def _ingest_audio(
    user_id: str,
    file_id: str,
    file_name: str,
    url: str,
    data: bytes,
    vectorstore: Any,
    role_id: str = "default",
) -> IngestResult:
    """音频：Whisper 转写 → 用 CLIP 文本编码器把转写文本写入 EchoDoc（modality=audio）。"""
    try:
        from embedding import whisper

        audio = await asyncio.to_thread(whisper.embed_audio, data)
        transcript = (audio.get("text") or "").strip()
    except Exception as e:
        log_exception(
            logger,
            "audio transcribe failed",
            exc=e,
            stage="ingest_audio",
            event="transcribe_error",
            file_id=file_id,
            url=url,
            audio_bytes=len(data or b""),
        )
        return IngestResult(False, file_id, error=f"Audio transcribe failed: {e}")

    logger.info(
        "audio transcribed",
        extra=merge_extra(
            stage="ingest_audio",
            event="transcribed",
            file_id=file_id,
            text_len=len(transcript),
        ),
    )
    if not transcript:
        return IngestResult(False, file_id, error="Empty audio transcript")

    try:
        vectors = await asyncio.to_thread(compute_text_embeddings, [transcript])
    except Exception as e:
        log_exception(
            logger,
            "audio transcript embedding failed",
            exc=e,
            stage="ingest_audio",
            event="embed_error",
            file_id=file_id,
            url=url,
        )
        return IngestResult(False, file_id, error=f"Audio embedding failed: {e}")

    metadata = {
        "fileId": file_id,
        "fileName": file_name,
        "userId": user_id,
        "roleId": role_id,
        "sourceUrl": url,
        "chunkIndex": 0,
        "totalChunks": 1,
        "modality": "audio",
    }
    try:
        vectorstore.add_texts(ids=[file_id], texts=[transcript], metadatas=[metadata], embeddings=vectors)
    except Exception as e:
        log_exception(
            logger,
            "vector store write failed",
            exc=e,
            stage="ingest_audio",
            event="vector_write_error",
            file_id=file_id,
            url=url,
        )
        return IngestResult(False, file_id, error=f"Vector store failed: {e}")
    return IngestResult(True, file_id, chunks=1, parsed_text=transcript, modality="audio")


async def _ingest_video(
    user_id: str,
    file_id: str,
    file_name: str,
    url: str,
    data: bytes,
    vectorstore: Any,
    role_id: str = "default",
) -> IngestResult:
    """视频：关键帧聚合成 CLIP 512 向量 + 文字描述，写入 EchoDoc（modality=video）。"""
    try:
        from embedding import video_mae

        video = await asyncio.to_thread(video_mae.embed_video, data)
        vector = video.get("embedding") or []
        description = (video.get("description") or "").strip()
    except Exception as e:
        log_exception(
            logger,
            "video parse failed",
            exc=e,
            stage="ingest_video",
            event="parse_error",
            file_id=file_id,
            url=url,
            video_bytes=len(data or b""),
        )
        return IngestResult(False, file_id, error=f"Video parse failed: {e}")

    logger.info(
        "video parsed",
        extra=merge_extra(
            stage="ingest_video",
            event="parsed",
            file_id=file_id,
            dim=len(vector),
            desc_len=len(description),
        ),
    )
    if not vector or not any(vector):
        return IngestResult(False, file_id, error="Empty video embedding")

    metadata = {
        "fileId": file_id,
        "fileName": file_name,
        "userId": user_id,
        "roleId": role_id,
        "sourceUrl": url,
        "chunkIndex": 0,
        "totalChunks": 1,
        "modality": "video",
    }
    try:
        vectorstore.add_texts(
            ids=[file_id], texts=[description or file_name or ""], metadatas=[metadata], embeddings=[vector]
        )
    except Exception as e:
        log_exception(
            logger,
            "vector store write failed",
            exc=e,
            stage="ingest_video",
            event="vector_write_error",
            file_id=file_id,
            url=url,
        )
        return IngestResult(False, file_id, error=f"Vector store failed: {e}")
    return IngestResult(True, file_id, chunks=1, parsed_text=description, modality="video")


# Legacy alias kept for any callers using the old synchronous signature.
def ingest_background(user_id: str, file_obj: dict[str, Any], embeddings: Any, vectorstore: Any) -> dict[str, Any]:
    import asyncio

    if not isinstance(embeddings, ChineseCLIPEmbeddings):
        embeddings = ChineseCLIPEmbeddings()
    result = asyncio.run(ingest_file(user_id, file_obj, embeddings, vectorstore))
    return {
        "success": result.success,
        "file_id": result.file_id,
        "chunks": result.chunks,
        "error": result.error,
    }