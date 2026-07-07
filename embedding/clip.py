"""CLIP 图像 embedding：复用原有 Chinese-CLIP，统一对外 512 维接口。"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from config.config import get_settings
from utils.request_context import merge_extra

logger = logging.getLogger(__name__)


def embed_images(images: Sequence[bytes | Any]) -> list[list[float]]:
    """批量图像向量化。失败时返回 dim 长度的零向量，保证调用方拿到对齐后的结果。"""
    from embedding import models as _m

    cfg = get_settings().embedding
    if not images:
        return []
    t0 = time.perf_counter()
    try:
        out = _m.compute_image_embeddings(list(images), device=cfg.device)
        logger.info(
            "clip.embed_images ok",
            extra=merge_extra(
                stage="clip_embed_images",
                event="ok",
                count=len(images),
                dim=len(out[0]) if out else 0,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return out
    except Exception as e:
        logger.warning(
            "clip.embed_images fallback (zero vectors)",
            extra=merge_extra(
                stage="clip_embed_images",
                event="fallback",
                count=len(images),
                dim=cfg.dim,
                error=str(e)[:200],
            ),
        )
        return [[0.0] * cfg.dim for _ in images]


def dim() -> int:
    return get_settings().embedding.dim