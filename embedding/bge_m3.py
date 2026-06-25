from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import torch
from sentence_transformers import SentenceTransformer

from config.config import get_settings

if TYPE_CHECKING:
    from config.config import BGE3Settings

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model: SentenceTransformer | None = None
_device: str | None = None


def _resolve_device(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_internal(device: str = "auto") -> tuple[SentenceTransformer, str]:
    global _model, _device
    settings: BGE3Settings = get_settings().multimodal.bge_m3
    target_device = _resolve_device(device or settings.device)
    with _model_lock:
        if _model is None:
            logger.info("Loading BGE-M3 model=%s on device=%s", settings.model_name, target_device)
            _model = SentenceTransformer(settings.model_name, device=target_device)
            _model.eval()
            _device = target_device
            logger.info("BGE-M3 model loaded")
    return _model, _device


def load_model(device: str = "auto") -> tuple[SentenceTransformer, str]:
    return _load_internal(device)


def warmup(device: str = "auto") -> None:
    model, _ = _load_internal(device)
    try:
        dummy = ["warmup"]
        model.encode(dummy, batch_size=1)
        logger.info("BGE-M3 warmup completed")
    except Exception as e:
        logger.warning("BGE-M3 warmup failed: %s", e)


def compute_embeddings(texts: list[str], device: str = "auto") -> list[list[float]]:
    if not texts:
        return []
    settings: BGE3Settings = get_settings().multimodal.bge_m3
    model, _ = _load_internal(device or settings.device)
    try:
        embeddings = model.encode(
            texts,
            batch_size=settings.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
    except Exception as e:
        logger.error("compute_embeddings failed: %s", e)
        raise


def compute_query_embedding(query: str, device: str = "auto") -> list[float]:
    vecs = compute_embeddings([query], device=device)
    return vecs[0] if vecs else []