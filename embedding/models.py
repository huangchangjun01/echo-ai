from __future__ import annotations

import io
import logging
import threading
import time
from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from utils.request_context import merge_extra

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model: ChineseCLIPModel | None = None
_processor: ChineseCLIPProcessor | None = None
_device: torch.device | None = None


def detect_device(preferred: str = "auto") -> torch.device:
    """Resolve inference device.

    Order: explicit preferred -> cuda -> mps -> cpu.
    """
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_internal(device: str = "auto") -> tuple[ChineseCLIPModel, ChineseCLIPProcessor, torch.device]:
    global _model, _processor, _device
    with _model_lock:
        if _model is None:
            target_device = detect_device(device)
            model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
            t0 = time.perf_counter()
            logger.info(
                "loading Chinese-CLIP",
                extra=merge_extra(
                    stage="clip_load",
                    event="start",
                    model=model_name,
                    device=str(target_device),
                ),
            )
            _model = ChineseCLIPModel.from_pretrained(model_name)
            _processor = ChineseCLIPProcessor.from_pretrained(model_name)
            _model.eval()
            _model.to(target_device)
            _device = target_device
            logger.info(
                "Chinese-CLIP model loaded",
                extra=merge_extra(
                    stage="clip_load",
                    event="ok",
                    model=model_name,
                    device=str(target_device),
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                ),
            )
        else:
            logger.debug(
                "Chinese-CLIP model cached",
                extra=merge_extra(stage="clip_load", event="cached", device=str(_device)),
            )
    return _model, _processor, _device


def load_model(device: str = "auto") -> tuple[ChineseCLIPModel, ChineseCLIPProcessor, torch.device]:
    return _load_internal(device)


def warmup(device: str = "auto", batch_size: int = 1) -> None:
    """Run a dummy forward pass so the first real request doesn't pay cold-start cost."""
    model, processor, target_device = _load_internal(device)
    try:
        with torch.no_grad():
            dummy_text = ["warmup"] * max(1, batch_size)
            inputs = processor(text=dummy_text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            features = _extract_text_features(model.get_text_features(**inputs), target_device)
            _ = _normalize(features)
        if target_device.type == "cuda":
            torch.cuda.synchronize()
        logger.info(
            "Embedding warmup completed",
            extra=merge_extra(stage="clip_warmup", event="ok", device=str(target_device)),
        )
    except Exception as e:  # warmup failures must not crash the service
        logger.warning(
            "Embedding warmup failed",
            extra=merge_extra(
                stage="clip_warmup",
                event="error",
                device=str(target_device),
                error=str(e)[:200],
            ),
        )


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _to_device(inputs: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in inputs.items()}


def _to_pil(image_input: bytes | Image.Image | str) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    raise TypeError(f"Unsupported image input type: {type(image_input)!r}")


def _extract_text_features(out: Any, target_device: torch.device) -> torch.Tensor:
    """Pull the projected text feature tensor out of whatever the model returned.

    transformers >=5 returns a `BaseModelOutputWithPooling` whose `pooler_output`
    is the projected embedding. Older versions returned a bare tensor.
    """
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    raise TypeError(f"Unexpected output from get_text_features: {type(out)!r}")


def _extract_image_features(out: Any, target_device: torch.device) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    raise TypeError(f"Unexpected output from get_image_features: {type(out)!r}")


def compute_text_embeddings(texts: Sequence[str], device: str = "auto") -> list[list[float]]:
    """Batched text embedding. Returns one L2-normalized vector per input string."""
    texts = list(texts)
    if not texts:
        return []
    model, processor, target_device = _load_internal(device)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = _to_device(inputs, target_device)
            features = _extract_text_features(model.get_text_features(**inputs), target_device)
            features = _normalize(features)
            out = features.cpu().tolist()
        logger.info(
            "text embedding ok",
            extra=merge_extra(
                stage="clip_text",
                event="ok",
                count=len(texts),
                dim=len(out[0]) if out else 0,
                device=str(target_device),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return out
    except Exception as e:
        logger.exception(
            "compute_text_embeddings failed",
            extra=merge_extra(
                stage="clip_text",
                event="error",
                count=len(texts),
                device=str(target_device),
                error=str(e)[:300],
            ),
        )
        raise


def compute_image_embeddings(
    images: Sequence[bytes | Image.Image | str],
    device: str = "auto",
) -> list[list[float]]:
    """Batched image embedding. Returns one L2-normalized vector per input image."""
    pil_images = [_to_pil(img) for img in images]
    if not pil_images:
        return []
    model, processor, target_device = _load_internal(device)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = _to_device(inputs, target_device)
            features = _extract_image_features(model.get_image_features(**inputs), target_device)
            features = _normalize(features)
            out = features.cpu().tolist()
        logger.info(
            "image embedding ok",
            extra=merge_extra(
                stage="clip_image",
                event="ok",
                count=len(pil_images),
                dim=len(out[0]) if out else 0,
                device=str(target_device),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return out
    except Exception as e:
        logger.exception(
            "compute_image_embeddings failed",
            extra=merge_extra(
                stage="clip_image",
                event="error",
                count=len(pil_images),
                device=str(target_device),
                error=str(e)[:300],
            ),
        )
        raise


# Backward-compat single-input helpers (kept so legacy call sites keep working).
def compute_embedding(image_bytes: bytes) -> list:
    if not isinstance(image_bytes, bytes):
        raise TypeError("compute_embedding expects raw image bytes")
    return compute_image_embeddings([image_bytes])[0]


def compute_text_embedding(text: str) -> list:
    return compute_text_embeddings([text])[0]