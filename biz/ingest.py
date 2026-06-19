from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.config import get_settings
from embedding.embeddings import ChineseCLIPEmbeddings
from embedding.models import compute_image_embeddings, compute_text_embeddings
from utils.downloader import DownloadError, download_file_async

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
# MIMEs we explicitly do NOT support (audio / video / archives / etc.).
REJECTED_BINARY_MIMES = {
    "video/mp4",
    "video/webm",
    "audio/mpeg",
    "audio/ogg",
    "audio/flac",
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


def _detect_mime(data: bytes) -> str:
    """Detect MIME via libmagic when available; fall back to a tiny magic-byte sniffer."""
    try:
        import magic  # type: ignore

        return magic.from_buffer(data, mime=True) or "application/octet-stream"
    except Exception:
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
            except Exception:
                # Allow lone invalid tail bytes (common in CJK when the cut falls mid-codepoint).
                bad = sample.decode(enc, errors="ignore").count("�")
                if bad <= max(1, len(sample) // 64):
                    return "text/plain"
        return "application/octet-stream"


def _decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return data.decode(enc)
        except Exception:
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
    except Exception:
        # Fallback: fixed-size windows.
        step = max(1, chunk_size - chunk_overlap)
        return [text[i : i + chunk_size] for i in range(0, max(1, len(text)), step)]


async def _download_with_retry(url: str) -> bytes:
    settings = get_settings().ingest
    last_exc: Exception | None = None
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max(1, settings.download_retries)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((DownloadError, asyncio.TimeoutError)),
        reraise=True,
    ):
        with attempt:
            try:
                return await download_file_async(url)
            except Exception as e:
                last_exc = e
                logger.warning("download attempt failed: %s", e)
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
) -> IngestResult:
    """Download, classify, embed, and persist a single file."""
    file_id = file_obj.get("fileId") or ""
    file_name = file_obj.get("fileName") or ""
    file_key = file_obj.get("fileKey")
    url = _resolve_url(file_key, file_obj.get("url"))
    if not url:
        return IngestResult(False, file_id, error="Missing URL")

    try:
        data = await _download_with_retry(url)
    except Exception as e:
        logger.error("download failed file_id=%s url=%s: %s", file_id, url, e)
        return IngestResult(False, file_id, error=f"Download failed: {e}")

    mime = _detect_mime(data)

    if mime in SUPPORTED_IMAGE_MIMES:
        return await _ingest_image(user_id, file_id, file_name, url, data, embeddings, vectorstore)

    if mime in REJECTED_BINARY_MIMES:
        return IngestResult(False, file_id, error=f"Unsupported binary content type: {mime}")

    if mime.startswith("text/") or mime in SUPPORTED_TEXT_MIMES:
        text = _decode_text(data)
        return await _ingest_text(user_id, file_id, file_name, url, text, embeddings, vectorstore)

    return IngestResult(
        False,
        file_id,
        error=f"Unsupported content type: {mime}",
    )


async def _ingest_text(
    user_id: str,
    file_id: str,
    file_name: str,
    url: str,
    text: str,
    embeddings: ChineseCLIPEmbeddings,
    vectorstore: Any,
) -> IngestResult:
    settings = get_settings().ingest
    if not settings.enable_chunking or len(text) <= settings.max_download_bytes // 1024:
        chunks = [text] if text else []
    else:
        chunks = _split_text(text, chunk_size=0, chunk_overlap=0)

    if not chunks:
        return IngestResult(False, file_id, error="Empty text content")

    try:
        vectors = await asyncio.to_thread(compute_text_embeddings, chunks)
    except Exception as e:
        logger.error("text embedding failed file_id=%s: %s", file_id, e)
        return IngestResult(False, file_id, error=f"Embedding failed: {e}")

    base_meta = {
        "fileId": file_id,
        "fileName": file_name,
        "userId": user_id,
        "sourceUrl": url,
        "totalChunks": len(chunks),
    }
    ids = [f"{file_id}:{i}" for i in range(len(chunks))]
    metadatas = [{**base_meta, "chunkIndex": i} for i in range(len(chunks))]
    try:
        vectorstore.add_texts(ids=ids, texts=chunks, metadatas=metadatas, embeddings=vectors)
    except Exception as e:
        logger.error("vector store write failed file_id=%s: %s", file_id, e)
        return IngestResult(False, file_id, error=f"Vector store failed: {e}")
    return IngestResult(True, file_id, chunks=len(chunks))


async def _ingest_image(
    user_id: str,
    file_id: str,
    file_name: str,
    url: str,
    data: bytes,
    embeddings: ChineseCLIPEmbeddings,
    vectorstore: Any,
) -> IngestResult:
    try:
        vectors = await asyncio.to_thread(compute_image_embeddings, [data])
    except Exception as e:
        logger.error("image embedding failed file_id=%s: %s", file_id, e)
        return IngestResult(False, file_id, error=f"Image embedding failed: {e}")

    metadata = {
        "fileId": file_id,
        "fileName": file_name,
        "userId": user_id,
        "sourceUrl": url,
        "chunkIndex": 0,
        "totalChunks": 1,
    }
    try:
        vectorstore.add_texts(ids=[file_id], texts=[file_name or ""], metadatas=[metadata], embeddings=vectors)
    except Exception as e:
        logger.error("vector store write failed file_id=%s: %s", file_id, e)
        return IngestResult(False, file_id, error=f"Vector store failed: {e}")
    return IngestResult(True, file_id, chunks=1)


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