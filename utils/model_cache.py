"""本地模型统一缓存目录管理：优先从指定缓存目录获取，取不到时再下载。

覆盖的本地模型：
- 文本模型：BGE-M3（sentence-transformers / HF hub）→ embedding.bge_m3
- 图片模型：Chinese-CLIP（transformers / HF hub）→ embedding.models
- 视频模型：VideoMAE 降级后复用 Chinese-CLIP 关键帧（无独立权重下载）
- 语音模型：Whisper（openai-whisper）→ embedding.whisper

缓存目录结构（由 `MODEL_CACHE_DIR` 指定时）：
    <MODEL_CACHE_DIR>/huggingface/hub   HF hub 缓存根（含 models--<org>--<name>）
    <MODEL_CACHE_DIR>/whisper           Whisper 权重目录（*.pt）

未指定时回退到各库默认位置（HF: ~/.cache/huggingface，whisper: ~/.cache/whisper），
保证已缓存的模型无需重复下载。

核心语义（对应需求「每次启动/使用从指定缓存目录中获取，取不到时再下载」）：
    1. 先定位本地快照（纯本地扫描，不发任何网络请求）；
    2. 命中 → 直接返回快照路径，后续以 local_files_only 加载，全程离线；
    3. 未命中且允许下载（MODEL_AUTO_DOWNLOAD=true）→ snapshot_download 拉取进缓存目录；
    4. 未命中且禁止下载 → 抛 ModelNotCachedError，由上层按各模型容错策略处理。
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from config.config import get_settings
from utils.request_context import log_exception, log_silent_failure, merge_extra

logger = logging.getLogger(__name__)


class ModelNotCachedError(Exception):
    """模型在缓存目录中未命中，且当前配置禁止联网下载。"""


def resolve_cache_base() -> Path | None:
    """解析模型缓存根目录；未配置（MODEL_CACHE_DIR 为空）返回 None（用库默认）。"""
    cfg = get_settings().model
    raw = (cfg.cache_dir or "").strip()
    if not raw:
        return None
    base = Path(raw).expanduser()
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log_silent_failure(
            logger,
            "model cache dir not writable, fallback to library default",
            exc=e,
            stage="model_cache",
            event="mkdir_error",
            cache_dir=raw,
        )
        return None
    return base


def resolve_hf_cache_root() -> str | None:
    """返回 HF hub 缓存根目录（含 models--<org>--<name>）；未指定时返回 None。

    与 huggingface_hub 的 `cache_dir` / transformers 的 `cache_dir` /
    sentence-transformers 的 `cache_folder` 语义一致，可直接透传给各加载器。
    """
    base = resolve_cache_base()
    if base is None:
        return None
    root = base / "huggingface" / "hub"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log_silent_failure(
            logger,
            "hf cache dir not writable, fallback to library default",
            exc=e,
            stage="model_cache",
            event="mkdir_error",
            cache_dir=str(root),
        )
        return None
    return str(root)


def resolve_whisper_root() -> str | None:
    """返回 Whisper 权重目录；未指定时返回 None（用 whisper 默认 ~/.cache/whisper）。"""
    base = resolve_cache_base()
    if base is None:
        return None
    root = base / "whisper"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log_silent_failure(
            logger,
            "whisper cache dir not writable, fallback to library default",
            exc=e,
            stage="model_cache",
            event="mkdir_error",
            cache_dir=str(root),
        )
        return None
    return str(root)


def apply_hf_env(endpoint: str | None = None, download_timeout: int | None = None) -> None:
    """把镜像端点 / 下载超时 / 缓存根固化到进程环境变量。

    必须在第一次触发 HF 请求之前调用，否则对应 httpx 客户端已经用旧 endpoint 起好。
    只使用 setdefault：尊重用户 shell 里显式设置的 HF_ENDPOINT / HF_HOME 等。
    """
    if endpoint:
        os.environ.setdefault("HF_ENDPOINT", endpoint)
    if download_timeout:
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(download_timeout))
    base = resolve_cache_base()
    if base is not None:
        # 让 transformers / sentence-transformers / huggingface_hub 全部落到统一缓存根
        os.environ.setdefault("HF_HOME", str(base / "huggingface"))
        os.environ.setdefault("HF_HUB_CACHE", str(base / "huggingface" / "hub"))


def _locate_cached_snapshot(model_id: str, cache_root: str | None) -> Path | None:
    """在 HF 缓存根中定位指定模型已缓存的快照；纯本地扫描，不触发网络请求。

    优先用 huggingface_hub.scan_cache_dir 的官方解析；异常时按经典磁盘布局兜底
    （refs/main → snapshots/<sha>，再退回最近的非空快照目录）。
    """
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir(cache_dir=cache_root)
        for repo in getattr(info, "repos", ()) or ():
            if getattr(repo, "repo_id", None) != model_id:
                continue
            revisions = sorted(
                getattr(repo, "revisions", ()) or (),
                key=lambda r: getattr(r, "last_modified", None) or 0,
                reverse=True,
            )
            for rev in revisions:
                snap = getattr(rev, "snapshot_path", None)
                if snap and Path(snap).is_dir() and any(Path(snap).iterdir()):
                    return Path(snap)
    except Exception as e:
        log_silent_failure(
            logger,
            "scan_cache_dir failed, fallback to manual cache layout check",
            exc=e,
            stage="model_cache",
            event="scan_error",
            model=model_id,
        )

    # ---- 手动兜底：经典 HF 缓存布局 ----
    if cache_root:
        repo_dir = Path(cache_root) / f"models--{model_id.replace('/', '--')}"
    else:
        repo_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    if not repo_dir.is_dir():
        return None
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    for ref_file in (repo_dir / "refs" / "main", repo_dir / ".refs" / "main"):
        try:
            sha = ref_file.read_text(encoding="utf-8").strip()
            if sha:
                snap = snapshots_dir / sha
                if snap.is_dir() and any(snap.iterdir()):
                    return snap
        except OSError:
            continue
    candidates = [d for d in snapshots_dir.iterdir() if d.is_dir() and any(d.iterdir())]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def ensure_hf_model(
    model_id: str,
    *,
    revision: str = "main",
    endpoint: str | None = None,
    download_timeout: int | None = None,
    allow_download: bool | None = None,
) -> Path:
    """确保 HF 模型快照存在于缓存目录，返回快照路径。

    - 缓存命中 → 直接返回（无网络）；
    - 未命中且允许下载（默认 MODEL_AUTO_DOWNLOAD）→ 下载进缓存目录；
    - 未命中且禁止下载 → 抛 ModelNotCachedError。
    """
    cfg = get_settings().model
    apply_hf_env(endpoint=endpoint, download_timeout=download_timeout)
    cache_root = resolve_hf_cache_root()

    snap = _locate_cached_snapshot(model_id, cache_root)
    if snap is not None:
        logger.info(
            "model cache hit, load locally",
            extra=merge_extra(
                stage="model_cache",
                event="hit",
                model=model_id,
                cache_dir=cache_root,
                snapshot=str(snap),
            ),
        )
        return snap

    if allow_download is None:
        allow_download = cfg.auto_download
    if not allow_download:
        raise ModelNotCachedError(
            f"model {model_id!r} not in cache dir and auto_download disabled "
            f"(cache_dir={cache_root or 'default'})"
        )

    logger.info(
        "model cache miss, start download",
        extra=merge_extra(
            stage="model_cache",
            event="download_start",
            model=model_id,
            endpoint=endpoint or os.environ.get("HF_ENDPOINT"),
            download_timeout=download_timeout,
            cache_dir=cache_root or "default",
        ),
    )
    t0 = time.perf_counter()
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_root,
            local_files_only=False,
            etag_timeout=float(download_timeout or 10),
        )
    except Exception as e:
        log_exception(
            logger,
            "model download failed",
            exc=e,
            level=logging.WARNING,
            stage="model_cache",
            event="download_error",
            model=model_id,
            endpoint=endpoint or os.environ.get("HF_ENDPOINT"),
            cache_dir=cache_root or "default",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        raise ModelNotCachedError(f"download model {model_id!r} failed: {e}") from e

    snap = Path(path)
    logger.info(
        "model downloaded to cache",
        extra=merge_extra(
            stage="model_cache",
            event="download_ok",
            model=model_id,
            cache_dir=cache_root or "default",
            snapshot=str(snap),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        ),
    )
    return snap
