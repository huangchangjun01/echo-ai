from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

import httpx

from config.config import get_settings
from utils.request_context import log_silent_failure

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Raised when a download cannot be completed safely."""


def _resolve_addresses(host: str) -> set[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise DownloadError(f"DNS resolution failed for host {host!r}: {e}") from e
    return {info[4][0] for info in infos}


def _ip_is_blocked(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def _is_safe_url(url: str, allowed_subdomains: Iterable[str]) -> None:
    """Validate URL host is allowed and resolves to non-internal IPs.

    - Scheme must be http or https.
    - Host must not be an IP literal that points to private/loopback ranges.
    - If `allowed_subdomains` is set, the host's rightmost labels must match one of them.
    - Otherwise, resolved IPs must not be in blocked ranges.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise DownloadError(f"Refusing non-http(s) URL: {url!r}")
    host = parsed.hostname
    if not host:
        raise DownloadError(f"URL has no host: {url!r}")

    allowed = [s.lower().lstrip(".") for s in allowed_subdomains if s]
    if allowed:
        host_lower = host.lower()
        if not any(host_lower == sub or host_lower.endswith("." + sub) for sub in allowed):
            raise DownloadError(f"Host {host!r} not in allowed subdomain list")

    # Resolve host -> IPs. If host is itself an IP literal, evaluate directly.
    try:
        ipaddress.ip_address(host)
        addresses = {host}
    except ValueError:
        try:
            addresses = _resolve_addresses(host)
        except DownloadError:
            if allowed:
                # Allowlist matched, but DNS failed: surface as download error.
                raise
            raise DownloadError(f"Could not resolve host {host!r}")

    blocked = [ip for ip in addresses if _ip_is_blocked(ip)]
    if blocked:
        raise DownloadError(f"Refusing URL pointing at blocked address(es): {blocked}")


async def download_file_async(
    url: str,
    *,
    max_bytes: int | None = None,
    timeout: float | None = None,
    allowed_subdomains: Iterable[str] | None = None,
    verify_ssl: bool = True,
) -> bytes:
    """Streamed async download with size guard and SSRF checks.

    verify_ssl=False 仅用于对象存储 CDN 子域的 HTTPS 证书可能与 host 不匹配的场景
    （如 Qiniu `*.clouddn.com` 子域）。其它场景必须保持 True。
    """
    settings = get_settings()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    allowed = list(allowed_subdomains) if allowed_subdomains is not None else settings.qiniu.allowed_subdomains
    if not allowed and settings.qiniu.base_url:
        # Auto-derive the allowed subdomain from the configured Qiniu base URL.
        try:
            qiniu_host = urlparse(settings.qiniu.base_url).hostname
            if qiniu_host:
                allowed = [qiniu_host]
        except Exception as e:
            log_silent_failure(
                logger,
                "auto-derive allowed subdomain from qiniu base_url failed (skip)",
                exc=e,
                stage="downloader_resolve",
                event="qiniu_base_parse_error",
                base_url=settings.qiniu.base_url,
            )

    _is_safe_url(url, allowed)

    max_bytes = max_bytes if max_bytes is not None else settings.ingest.max_download_bytes
    timeout_s = timeout if timeout is not None else settings.ingest.download_timeout_seconds

    chunks: list[bytes] = []
    total = 0
    timeout_cfg = httpx.Timeout(timeout_s, connect=10.0)
    # http2=False：避免 Qiniu CDN 触发 HTTP/2 negotiation 后返回 421 Misdirected Request
    async with httpx.AsyncClient(timeout=timeout_cfg, follow_redirects=True, verify=verify_ssl, http2=False) as client:
        async with client.stream("GET", url) as resp:
            if resp.status_code >= 400:
                raise DownloadError(f"HTTP {resp.status_code} for {url}")
            content_length = resp.headers.get("content-length")
            if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                raise DownloadError(
                    f"Remote content-length {content_length} exceeds max {max_bytes}"
                )
            async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise DownloadError(f"Download exceeded max bytes {max_bytes}")
                chunks.append(chunk)
    return b"".join(chunks)


def download_file(url: str, *, max_bytes: int | None = None, timeout: float | None = None, verify_ssl: bool = True) -> bytes:
    """Synchronous wrapper for legacy call sites and tests."""
    return asyncio.run(download_file_async(url, max_bytes=max_bytes, timeout=timeout, verify_ssl=verify_ssl))


async def fetch_source_bytes(
    file_key: str | None,
    url: str | None,
    *,
    max_bytes: int | None = None,
    timeout: float | None = None,
) -> bytes:
    """下载源文件字节：优先 fileKey 直连七牛私有空间（权威、稳定），失败再退回前端 url。

    为什么优先 fileKey：前端传来的 url 可能是 dev server / SPA fallback / 404 页，
    直接下载会把 HTML 当成源文件内容写进记忆 md（本次缺陷根因）。用 fileKey
    直连对象存储可根治，url 仅作无 fileKey 时的兜底。text/image/audio/video
    全部走本函数，保持解析链路对"错误 url"免疫。
    """
    # 1) fileKey 优先：直连七牛私有空间（download_object_bytes 已处理 CDN 421 / 证书链问题）
    if file_key:
        try:
            from storage.qiniu_client import download_object_bytes

            kwargs: dict[str, Any] = {}
            if max_bytes is not None:
                kwargs["max_bytes"] = max_bytes
            if timeout is not None:
                kwargs["timeout"] = timeout
            return await download_object_bytes(file_key, **kwargs)
        except Exception as e:
            log_silent_failure(
                logger,
                "fileKey 下载失败，回退前端 url",
                exc=e,
                stage="downloader_source",
                event="filekey_fallback",
                file_key=file_key,
            )

    # 2) url 兜底
    if url:
        return await download_file_async(url, max_bytes=max_bytes, timeout=timeout)

    raise DownloadError("fetch_source_bytes: 既无 file_key 也无 url")