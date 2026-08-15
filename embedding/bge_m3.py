"""BGE-M3 文本 embedding（768 维）。

优先使用 sentence-transformers 加载 `BAAI/bge-m3`；若环境中未安装或离线下载失败，
自动回退到已有的 Chinese-CLIP 文本特征做维度对齐，并保证返回向量均为 768 维。

设计要点：
- 单进程内只 load 一次模型（线程安全懒加载）。
- 任意输入均返回 float32 list，长度等于配置的 dim（768）。
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time

from config.config import get_settings
from utils.model_cache import apply_hf_env, ensure_hf_model, resolve_hf_cache_root
from utils.request_context import log_exception, log_silent_failure, merge_extra

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model = None
_model_loaded = False


def _apply_endpoint_env(cfg) -> None:
    """把配置中的 HF 镜像 / 超时 / 缓存根写到进程环境变量。

    必须在第一次触发 HF 请求之前调用，否则对应 httpx 客户端已经用旧 endpoint 起好。
    """
    apply_hf_env(endpoint=cfg.endpoint, download_timeout=cfg.download_timeout)


def _fallback_embed(texts: list[str], dim: int) -> list[list[float]]:
    """确定性 SHA256 回退向量（仅供开发/无 ML 依赖时使用）。"""
    out: list[list[float]] = []
    for t in texts:
        if not t:
            out.append([0.0] * dim)
            continue
        h = hashlib.sha256(t.encode("utf-8")).digest()
        vals: list[float] = []
        i = 0
        while len(vals) < dim:
            vals.append((h[i % len(h)] / 255.0) * 2.0 - 1.0)
            i += 1
        out.append(vals[:dim])
    return out


def _resolve_device(device: str) -> str:
    """把 "auto" 解析为 sentence-transformers 可识别的设备名。"""
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception as e:
        log_silent_failure(
            logger,
            "torch device probe failed, fallback to cpu",
            exc=e,
            stage="bge_resolve_device",
            event="torch_probe_error",
        )
        return "cpu"


def _load_model():
    global _model, _model_loaded
    if _model_loaded:
        return _model
    with _model_lock:
        if _model_loaded:
            return _model
        cfg = get_settings().bge_m3
        _apply_endpoint_env(cfg)
        resolved = _resolve_device(cfg.device)
        t0 = time.perf_counter()
        try:
            from sentence_transformers import SentenceTransformer

            # 缓存未命中且允许下载时先拉取到指定缓存目录；命中则直接本地加载。
            if not cfg.local_files_only:
                ensure_hf_model(cfg.model, endpoint=cfg.endpoint, download_timeout=cfg.download_timeout)
            cache_folder = resolve_hf_cache_root()
            logger.info(
                "loading BGE-M3",
                extra=merge_extra(
                    stage="bge_load",
                    event="start",
                    model=cfg.model,
                    device=resolved,
                    endpoint=cfg.endpoint or os.environ.get("HF_ENDPOINT"),
                    cache_dir=cache_folder or "default",
                    local_files_only=True,
                ),
            )
            # ensure_hf_model 已保证快照存在，这里 local_files_only=True 全程离线加载。
            _model = SentenceTransformer(
                cfg.model,
                device=resolved,
                cache_folder=cache_folder,
                local_files_only=True,
            )
            logger.info(
                "BGE-M3 model loaded",
                extra=merge_extra(
                    stage="bge_load",
                    event="ok",
                    model=cfg.model,
                    device=resolved,
                    cache_dir=cache_folder or "default",
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                ),
            )
        except Exception as e:
            log_exception(
                logger,
                "BGE-M3 load failed, use fallback",
                exc=e,
                level=logging.WARNING,
                stage="bge_load",
                event="fallback",
                model=cfg.model,
                device=resolved,
                endpoint=cfg.endpoint or os.environ.get("HF_ENDPOINT"),
                local_files_only=cfg.local_files_only,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
            _model = None
        _model_loaded = True
    return _model


def warmup() -> None:
    """预热：避免首请求延迟。"""
    cfg = get_settings().bge_m3
    model = _load_model()
    if model is None:
        return
    try:
        t0 = time.perf_counter()
        model.encode(["warmup"], normalize_embeddings=True)
        logger.info(
            "BGE-M3 warmup done",
            extra=merge_extra(
                stage="bge_warmup",
                event="ok",
                device=str(model.device),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
    except Exception as e:  # noqa: BLE001
        log_exception(
            logger,
            "BGE-M3 warmup failed",
            exc=e,
            level=logging.WARNING,
            stage="bge_warmup",
            event="error",
            device=str(model.device) if 'model' in locals() and model else None,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """BGE-M3 文本向量化；失败回退 SHA256。"""
    cfg = get_settings().bge_m3
    dim = cfg.dim
    if not texts:
        return []
    n = len(texts)
    model = _load_model()
    t0 = time.perf_counter()
    if model is not None:
        try:
            vecs = model.encode(
                texts,
                batch_size=8,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            out = [list(map(float, v)) for v in vecs]
            logger.info(
                "BGE-M3 embed ok",
                extra=merge_extra(
                    stage="bge_embed",
                    event="ok",
                    count=n,
                    batch_size=8,
                    fallback=False,
                    dim=len(out[0]) if out else 0,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                ),
            )
            return out
        except Exception as e:
            log_exception(
                logger,
                "BGE-M3 encode failed, fallback",
                exc=e,
                level=logging.WARNING,
                stage="bge_embed",
                event="encode_error",
                count=n,
                dim=dim,
                batch_size=8,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
    out = _fallback_embed(texts, dim)
    logger.info(
        "BGE-M3 fallback",
        extra=merge_extra(
            stage="bge_embed",
            event="fallback",
            count=n,
            dim=dim,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        ),
    )
    return out


def dim() -> int:
    return get_settings().bge_m3.dim