import os
import tempfile
import logging
from typing import Any, Dict, Optional

from config.config import QINIU_BASE_URL

logger = logging.getLogger(__name__)

# Local temp directory for downloaded files
TEMP_FILE_DIR = tempfile.gettempdir()

# File extension categories
TEXT_EXTS = frozenset({'.txt', '.md', '.csv', '.json', '.log', '.py', '.mdown', '.markdown'})
IMAGE_EXTS = frozenset({'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'})
AUDIO_EXTS = frozenset({'.mp3', '.wav', '.m4a', '.flac', '.aac'})
VIDEO_EXTS = frozenset({'.mp4', '.mov', '.avi', '.mkv'})

try:
    from utils import download_file
except Exception:
    import requests

    def download_file(url: str) -> bytes:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content


def _lazy_init_embedding_model(file_id: str):
    """Lazily initialize the embedding model if not provided."""
    try:
        from embedding.embeddings import ChineseCLIPEmbeddings
        model = ChineseCLIPEmbeddings()
        logger.info(f"[ingest_background] Lazily initialized embedding model for file_id={file_id}")
        return model
    except Exception as e:
        logger.error(f"[ingest_background] Failed to initialize embedding model: {e}")
        return None


def _decode_text(data: bytes, default: str) -> str:
    """Try decoding bytes as UTF-8, then GBK, returning default on failure."""
    for enc in ('utf-8', 'gbk'):
        try:
            return data.decode(enc)
        except Exception:
            pass
    return default


def _build_metadata(file_id: str, file_name: str, user_id: str, url: str) -> Dict[str, Any]:
    """Build metadata dict for vector store."""
    return {
        'fileId': file_id,
        'fileName': file_name,
        'userId': user_id,
        'sourceUrl': url,
    }


def _save_temp_file(data: bytes, file_name: str) -> Optional[str]:
    """Save bytes to a temp file, returning the path or None on failure."""
    try:
        _, ext = os.path.splitext(file_name or "")
        ext = ext.lower()
        fd, path = tempfile.mkstemp(suffix=ext, dir=TEMP_FILE_DIR)
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        logger.info(f"[ingest_background] Saved temp file: {path}")
        return path
    except Exception as e:
        logger.warning(f"[ingest_background] Failed to save temp file: {e}")
        return None


def _delete_temp_file(path: str) -> None:
    """Delete a temp file, ignoring errors."""
    try:
        os.remove(path)
        logger.info(f"[ingest_background] Deleted temp file: {path}")
    except Exception as e:
        logger.warning(f"[ingest_background] Failed to delete temp file {path}: {e}")


def _resolve_url(file_key: str, url: str) -> str:
    """Resolve final URL from file_key or existing url."""
    if url:
        return url
    if file_key and QINIU_BASE_URL:
        return QINIU_BASE_URL.rstrip('/') + '/' + file_key.lstrip('/')
    return ""


def _compute_embeddings(file_ext: str, data: bytes, text: str, embedding_model: Any) -> Optional[list]:
    """Compute embeddings based on file type, returning None on failure."""
    try:
        if embedding_model is None:
            return None
        if file_ext in TEXT_EXTS:
            return embedding_model.embed_documents([text])
        if file_ext in IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS | {'.unknown'}:
            return embedding_model.embed_images([data])
        return None
    except Exception as e:
        logger.error(f"[ingest_background] Embedding generation failed: {e}")
        return None


def ingest_background(user_id: str, file_obj: Dict[str, Any], embedding_model: Any, vectorstore: Any):
    """Background worker: download file, compute embedding, and store into vector DB.

    This function is designed to be scheduled with FastAPI BackgroundTasks and thus should be
    synchronous (not async). It expects an `embedding_model` that implements
    `embed_documents` and `embed_images` similar to `ChineseCLIPEmbeddings`.

    Returns:
        Dict with 'success' status and message, or 'error' status with error details.
    """
    file_id = file_obj.get('fileId')
    file_name = file_obj.get('fileName')
    file_key = file_obj.get('fileKey')
    url = _resolve_url(file_key, file_obj.get('url'))

    if not url:
        logger.error(f"[ingest_background] Missing URL for file_id={file_id}, file_key={file_key}")
        return {"success": False, "error": "Missing URL", "file_id": file_id}

    ext = os.path.splitext(file_name or "")[1].lower()

    try:
        # Download
        try:
            data = download_file(url)
            logger.info(f"[ingest_background] Downloaded {len(data)} bytes for file_id={file_id}")
        except Exception as e:
            logger.error(f"[ingest_background] Failed to download file_id={file_id}, url={url}: {e}")
            return {"success": False, "error": f"Download failed: {str(e)}", "file_id": file_id}

        # Save temp file for cleanup tracking
        local_file_path = _save_temp_file(data, file_name)

        # Resolve embedding model
        model = embedding_model or _lazy_init_embedding_model(file_id)

        # Compute text content and embeddings based on file type
        text_to_store = _decode_text(data, file_name or "") if ext in TEXT_EXTS else file_name or ""
        embeddings = _compute_embeddings(ext, data, text_to_store, model)

        # Persist to vector store
        if vectorstore is not None and embeddings is not None:
            try:
                metadata = _build_metadata(file_id, file_name, user_id, url)
                vectorstore.add_texts(ids=[file_id], texts=[text_to_store], metadatas=[metadata], embeddings=embeddings)
                logger.info(f"[ingest_background] Stored vector for file_id={file_id}")
            except Exception as e:
                logger.error(f"[ingest_background] Vector store failed for file_id={file_id}: {e}")
                return {"success": False, "error": f"Vector store failed: {str(e)}", "file_id": file_id}

        # Cleanup
        if local_file_path:
            _delete_temp_file(local_file_path)

        logger.info(f"[ingest_background] Successfully processed file_id={file_id}")
        return {"success": True, "file_id": file_id}

    except Exception as e:
        logger.error(f"[ingest_background] Unexpected error: {e}")
        return {"success": False, "error": f"Unexpected error: {str(e)}", "file_id": file_id}