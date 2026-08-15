from __future__ import annotations

import hashlib
import io
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.embeddings import Embeddings

from config.config import get_settings
from embedding import models as _repo_models
from utils.request_context import log_exception, log_silent_failure

logger = logging.getLogger(__name__)


def _sha256_fallback(texts: list[str], dim: int) -> list[list[float]]:
    """Deterministic SHA256-based fallback used only when both CLIP and sentence-transformers are unavailable."""
    vecs: list[list[float]] = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        vals: list[float] = []
        i = 0
        while len(vals) < dim:
            b = h[i % len(h)]
            vals.append((b / 255.0) * 2.0 - 1.0)
            i += 1
        vecs.append(vals[:dim])
    return vecs


def _image_fallback(images: Sequence[bytes | Any], dim: int) -> list[list[float]]:
    vecs: list[list[float]] = []
    for img in images:
        if isinstance(img, bytes):
            b = img
        else:
            try:

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b = buf.getvalue()
            except Exception as e:
                log_silent_failure(
                    logger,
                    "image fallback: save to PNG failed (hash str instead)",
                    exc=e,
                    stage="clip_embed_image_fallback",
                    event="png_save_error",
                )
                b = str(img).encode("utf-8")
        h = hashlib.sha256(b).digest()
        vals: list[float] = []
        i = 0
        while len(vals) < dim:
            byte = h[i % len(h)]
            vals.append((byte / 255.0) * 2.0 - 1.0)
            i += 1
        vecs.append(vals[:dim])
    return vecs


class ChineseCLIPEmbeddings(Embeddings):
    """LangChain Embeddings wrapper around Chinese-CLIP.

    Uses batched `compute_text_embeddings` / `compute_image_embeddings` from
    `embedding.models`. Falls back to sentence-transformers for text, then to a
    deterministic SHA256 vector for environments without ML deps (development only).
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        settings = get_settings().embedding
        self.dim = settings.dim
        self.device = device or settings.device
        self.model_name = model_name or settings.model_name
        self._st = None

        # Prefer the project's CLIP batched functions when available.
        if hasattr(_repo_models, "compute_text_embeddings"):
            self._embed_fn = lambda texts: _repo_models.compute_text_embeddings(list(texts), device=self.device)
        else:
            try:
                from sentence_transformers import SentenceTransformer
                from utils.model_cache import resolve_hf_cache_root

                self._st = SentenceTransformer(
                    self.model_name or "sentence-transformers/all-MiniLM-L6-v2",
                    device=self.device,
                    cache_folder=resolve_hf_cache_root(),
                )
                self._embed_fn = lambda texts: [list(map(float, v)) for v in self._st.encode(list(texts))]
            except Exception as e:
                log_silent_failure(
                    logger,
                    "sentence_transformers unavailable, use SHA256 fallback",
                    exc=e,
                    stage="clip_embed_init",
                    event="st_load_error",
                    model=self.model_name,
                    device=self.device,
                )
                self._embed_fn = lambda texts: _sha256_fallback(list(texts), self.dim)

        if hasattr(_repo_models, "compute_image_embeddings"):
            self._image_embed_fn = lambda images: _repo_models.compute_image_embeddings(list(images), device=self.device)
        else:
            self._image_embed_fn = lambda images: _image_fallback(list(images), self.dim)

    def warmup(self) -> None:
        if hasattr(_repo_models, "warmup"):
            try:
                _repo_models.warmup(device=self.device)
            except Exception as e:
                log_exception(
                    logger,
                    "embedding warmup failed",
                    exc=e,
                    level=logging.WARNING,
                    stage="embed_warmup",
                    event="error",
                    device=self.device,
                    model=self.model_name,
                )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        result = self._embed_fn(texts)
        return self._normalize_batch(result, expected=len(texts))

    def embed_query(self, text: str) -> list[float]:
        vecs = self.embed_documents([text])
        return vecs[0] if vecs else []

    def embed_images(self, images: Sequence[Any]) -> list[list[float]]:
        images = list(images)
        if not images:
            return []
        result = self._image_embed_fn(images)
        return self._normalize_batch(result, expected=len(images))

    def embed_image(self, image: Any) -> list[float]:
        vecs = self.embed_images([image])
        return vecs[0] if vecs else []

    def _normalize_batch(self, res: Any, expected: int) -> list[list[float]]:
        if res is None:
            return [[0.0] * self.dim for _ in range(expected)]
        try:
            res_list = res.tolist() if hasattr(res, "tolist") else res
        except Exception as e:
            log_silent_failure(
                logger,
                "res.tolist() failed (use raw)",
                exc=e,
                stage="clip_normalize_batch",
                event="res_tolist_error",
                expected=expected,
            )
            res_list = res

        if isinstance(res_list, (list, tuple)) and res_list and all(isinstance(x, (int, float)) for x in res_list):
            return [[float(x) for x in res_list]]

        out: list[list[float]] = []
        for item in res_list or []:
            if item is None:
                out.append([0.0] * self.dim)
                continue
            try:
                item_list = item.tolist() if hasattr(item, "tolist") else list(item)
            except Exception as e:
                log_silent_failure(
                    logger,
                    "item.tolist() failed (skip)",
                    exc=e,
                    stage="clip_normalize_batch",
                    event="item_tolist_error",
                    expected=expected,
                )
                continue
            if all(isinstance(x, (int, float)) for x in item_list):
                out.append([float(x) for x in item_list])
        while len(out) < expected:
            out.append([0.0] * self.dim)
        return out[:expected]