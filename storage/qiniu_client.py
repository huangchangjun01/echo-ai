"""七牛云对象存储客户端（用于回忆记忆的 md 文件直传/删除与源文件下载）。

注意：与 `utils.downloader.py` 配合使用：
- 下载走 downloader（带 SSRF 白名单 / 字节上限）
- 上传 / 删除走本客户端（需 AK/SK 写权限）
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

try:
    from qiniu import Auth, BucketManager, put_data  # type: ignore
    _QINIU_AVAILABLE = True
except Exception:  # noqa: BLE001
    Auth = None  # type: ignore
    BucketManager = None  # type: ignore
    put_data = None  # type: ignore
    _QINIU_AVAILABLE = False

from config.config import get_settings
from utils.request_context import log_exception, log_silent_failure, merge_extra

logger = logging.getLogger(__name__)

_singleton_lock = threading.Lock()
_auth: Auth | None = None
_bucket_mgr: BucketManager | None = None


def _ensure_credentials() -> tuple[Any, str]:
    """按需懒构造 Auth 单例，校验凭证齐备。"""
    global _auth, _bucket_mgr
    if not _QINIU_AVAILABLE:
        raise RuntimeError("qiniu SDK 未安装，请 pip install qiniu")
    if _auth is None:
        with _singleton_lock:
            if _auth is None:
                cfg = get_settings().qiniu
                if not cfg.access_key or not cfg.secret_key or not cfg.bucket_name:
                    raise RuntimeError(
                        "QINIU_ACCESS_KEY / QINIU_SECRET_KEY / QINIU_BUCKET_NAME 未配置，"
                        "回忆记忆的 md 直传/删除不可用"
                    )
                _auth = Auth(cfg.access_key, cfg.secret_key)  # type: ignore[misc]
                _bucket_mgr = BucketManager(_auth)  # type: ignore[misc]
    return _auth, get_settings().qiniu.bucket_name


def upload_bytes(key: str, data: bytes, mime_type: str = "application/octet-stream") -> None:
    """上传字节到七牛（同步，内部已是简单 PUT 调用）。"""
    try:
        auth, bucket = _ensure_credentials()
        token = auth.upload_token(bucket, key)
        ret, info = put_data(token, key, data, mime_type=mime_type)
        if info.status_code != 200:
            raise RuntimeError(f"qiniu upload failed status={info.status_code} ret={ret}")
        logger.info(
            "qiniu upload ok",
            extra=merge_extra(stage="qiniu_upload", event="ok", key=key, bytes=len(data)),
        )
    except Exception as e:
        log_exception(
            logger,
            "qiniu upload failed",
            exc=e,
            level=logging.ERROR,
            stage="qiniu_upload",
            event="error",
            key=key,
            bytes=len(data) if data else 0,
        )
        raise


def delete_object(key: str) -> None:
    """删除单个对象。"""
    try:
        auth, bucket = _ensure_credentials()
        # 重新构造 BucketManager 以保证 _bucket_mgr 已就绪
        global _bucket_mgr
        if _bucket_mgr is None:
            _bucket_mgr = BucketManager(auth)  # type: ignore[misc]
        ret, info = _bucket_mgr.delete(bucket, key)
        if info.status_code not in (200, 204, 612):  # 612 = 记录不存在，幂等
            raise RuntimeError(f"qiniu delete failed status={info.status_code} ret={ret}")
        logger.info(
            "qiniu delete ok",
            extra=merge_extra(stage="qiniu_delete", event="ok", key=key),
        )
    except Exception as e:
        log_exception(
            logger,
            "qiniu delete failed",
            exc=e,
            level=logging.ERROR,
            stage="qiniu_delete",
            event="error",
            key=key,
        )
        raise


def _has_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def make_private_url(key: str, expires_seconds: int = 3600) -> str:
    """签发对象存储私有空间临时访问 URL（给本服务内下载 md 用）。

    注意：本函数仅复用与 echo-core 一致的 AK/SK/Bucket，URL 走 Qiniu 私有空间下载约定。
    """
    auth, _bucket = _ensure_credentials()
    base = (get_settings().qiniu.base_url or "").strip()
    if base.startswith("http://") or base.startswith("https://"):
        domain = base
    else:
        domain = "https://" + base
    full_url = f"{domain.rstrip('/')}/{key.lstrip('/')}"
    # qiniu SDK private_download_url(url, expires=...)：传入完整 URL
    return auth.private_download_url(full_url, expires=expires_seconds)


def _make_private_url_http(key: str, expires_seconds: int = 3600) -> str:
    """下载场景专用：用 http:// 而非 https:// 拼域名，避免 Qiniu CDN https 证书链问题。

    视觉/多模态解析链路（image/audio/video）都走这个，与 utils.downloader 默认补 http:// 保持一致。
    """
    auth, _bucket = _ensure_credentials()
    base = (get_settings().qiniu.base_url or "").strip()
    if not base:
        raise RuntimeError("QINIU_BASE_URL 未配置")
    if base.startswith("http://") or base.startswith("https://"):
        domain = base
    else:
        domain = "http://" + base
    full_url = f"{domain.rstrip('/')}/{key.lstrip('/')}"
    return auth.private_download_url(full_url, expires=expires_seconds)


async def download_object_bytes(
    key: str,
    *,
    expires_seconds: int = 3600,
    timeout: float = 30.0,
    max_bytes: int = 50 * 1024 * 1024,
) -> bytes:
    """从七牛云私有空间下载指定 key 的对象 bytes。

    与 `utils.downloader.download_file_async` 的区别：
    - 不走子域名白名单校验（domain 是自家 Qiniu，可信）
    - 不补 scheme（强制走 http://，避开 https 证书链问题）
    - verify_ssl=False（兼容 CDN 子域证书不匹配）
    - http2=False（避免 Qiniu CDN 触发 421 Misdirected Request）

    推荐用法：
        多模态解析链路（image/audio/video）应当用 fileKey 调用本函数，
        **不要相信前端的 url 字段**——它可能是 dev server、404 页面、或 Vite SPA fallback。
    """
    import httpx

    from utils.downloader import DownloadError

    url = _make_private_url_http(key, expires_seconds=expires_seconds)
    timeout_cfg = httpx.Timeout(timeout, connect=10.0)
    chunks: list[bytes] = []
    total = 0
    try:
        async with httpx.AsyncClient(
            timeout=timeout_cfg,
            follow_redirects=True,
            verify=False,
            http2=False,
        ) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise DownloadError(f"qiniu GET {key} -> HTTP {resp.status_code}")
                async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise DownloadError(
                            f"qiniu GET {key} exceeded max {max_bytes} bytes"
                        )
                    chunks.append(chunk)
        logger.info(
            "qiniu download ok",
            extra=merge_extra(
                stage="qiniu_download",
                event="ok",
                key=key,
                bytes=total,
            ),
        )
        return b"".join(chunks)
    except Exception as e:
        log_exception(
            logger,
            "qiniu download failed",
            exc=e,
            level=logging.WARNING,
            stage="qiniu_download",
            event="error",
            key=key,
        )
        raise