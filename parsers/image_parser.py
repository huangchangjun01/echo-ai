"""图片解析器：调用视觉模型（OpenAI 兼容多模态 image_url）做细节描述。

设计要点（2026-07-27 修复）：
- 解析链路不再相信前端传来的 url 字段。前端传过来的 sourceUrl 可能是错的
  （dev server URL / SPA fallback / 404 页等），导致 LLM 拉图拿到 HTML。
- 改为用 fileKey 走 storage.qiniu_client.download_object_bytes 直接从七牛云私有空间下载。
- 下载后用 magic bytes 校验确实是合法图片（拦截 HTML/JSON/二进制噪声），
  再 base64 编码为 data URL 喂给视觉 LLM。LLM 服务端不再访问任何外网 URL。
- url 字段仅作为兜底（无 fileKey 时）。
"""

from __future__ import annotations

import base64
import logging

from llm.client import get_llm_client
from storage.qiniu_client import download_object_bytes
from utils.downloader import download_file_async
from utils.request_context import log_exception, log_silent_failure, merge_extra
from .base import ParsedFile, ParsedChunk

logger = logging.getLogger(__name__)

# 多模态单图上限 ~10MB（OpenAI / MiniMax 等都用这个量级）
_MAX_IMAGE_BYTES = 10 * 1024 * 1024

_VISION_PROMPT = (
    "你是一名视觉理解助手。请用 2~4 段中文详细描述这张图片：\n"
    "- 主体（人物/物体/场景）\n"
    "- 关键文字 / OCR 内容（如有）\n"
    "- 空间关系、显著颜色、视觉氛围\n"
    "- 用户视角下值得记忆的细节（如人物表情、地标、特殊物品）\n"
    "注意：只描述图片中实际可见的内容，不要凭空猜测；用一段连贯文字返回。"
)


def _normalize_image_url(url: str) -> str:
    """给裸域名 URL 补 scheme，避免被多模态 LLM 拒绝。"""
    if not url:
        return url
    lowered = url.lower()
    if lowered.startswith(("http://", "https://", "data:")):
        return url
    return f"http://{url}"


def _guess_image_mime(data: bytes) -> str | None:
    """用 magic bytes 判定 MIME；不匹配返回 None。

    用于拦截 LLM 服务端拉到 HTML（Vite SPA fallback）/ JSON 错误页时被误当图片。
    """
    if not data or len(data) < 12:
        return None
    head = data[:12]
    # WebP: 'RIFF' + 4 字节 size + 'WEBP'
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head.startswith(b"BM"):
        return "image/bmp"
    return None


async def _fetch_image_bytes(file_key: str | None, url: str | None) -> tuple[bytes, str, str] | None:
    """解析优先级：
    1. fileKey → 七牛云 SDK 下载（权威、稳定）
    2. url 字段兜底（httpx 下载，不走 SSRF 白名单）
    返回 (bytes, mime, source) 或 None。
    """
    # 1) fileKey 优先：直接走 Qiniu SDK
    if file_key:
        try:
            data = await download_object_bytes(
                file_key,
                timeout=30,
                max_bytes=_MAX_IMAGE_BYTES,
            )
            mime = _guess_image_mime(data)
            if mime:
                return data, mime, "qiniu_filekey"
            log_silent_failure(
                logger,
                "qiniu download returned non-image bytes",
                exc=None,
                stage="parser_image",
                event="qiniu_not_image",
                file_key=file_key,
                bytes=len(data),
                head_ascii=data[:32].decode("ascii", errors="replace"),
            )
        except Exception as e:
            log_exception(
                logger,
                "qiniu download failed (will try url fallback)",
                exc=e,
                level=logging.WARNING,
                stage="parser_image",
                event="qiniu_download_error",
                file_key=file_key,
            )

    # 2) url 字段兜底
    if url:
        norm_url = _normalize_image_url(url)
        try:
            data = await download_file_async(
                norm_url,
                max_bytes=_MAX_IMAGE_BYTES,
                timeout=30,
            )
            mime = _guess_image_mime(data)
            if mime:
                return data, mime, "url_fallback"
            log_silent_failure(
                logger,
                "url fallback returned non-image bytes",
                exc=None,
                stage="parser_image",
                event="url_not_image",
                url=norm_url,
                bytes=len(data),
                head_ascii=data[:32].decode("ascii", errors="replace"),
            )
        except Exception as e:
            log_exception(
                logger,
                "url fallback download failed",
                exc=e,
                level=logging.WARNING,
                stage="parser_image",
                event="url_download_error",
                url=url,
            )

    return None


async def _vision_describe_data_url(data_url: str) -> str:
    """data URL → 视觉 LLM 描述。data_url 已包含合法图片 mime + base64 内容。"""
    client = get_llm_client()
    messages = [
        {"role": "system", "content": _VISION_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图片。"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    try:
        resp = await client.chat(messages, max_tokens=600, temperature=0.4)
        try:
            return (resp["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return ""
    except Exception as e:
        log_exception(
            logger,
            "vision describe failed",
            exc=e,
            level=logging.WARNING,
            stage="parser_image",
            event="vision_error",
        )
        return ""


async def parse_image(file_key: str, file_name: str, url: str | None) -> ParsedFile:
    if not file_key and not url:
        return ParsedFile(modality="image", meta={"error": "no_key_or_url"})

    # 1) 解析图片内容（fileKey 优先走 Qiniu SDK，url 兜底）
    fetched = await _fetch_image_bytes(file_key or None, url)
    if fetched is None:
        return ParsedFile(
            modality="image",
            meta={"error": "fetch_failed", "file_key": file_key, "url": url},
        )
    img_bytes, mime, source = fetched

    # 2) base64 编码为 data URL，喂给视觉 LLM（LLM 服务端不再访问外网）
    b64 = base64.b64encode(img_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    desc = await _vision_describe_data_url(data_url)

    if not desc:
        return ParsedFile(
            modality="image",
            meta={"error": "vision_failed", "source": source, "file_key": file_key, "url": url},
        )

    logger.info(
        "image parsed",
        extra=merge_extra(
            stage="parser_image",
            event="ok",
            file_name=file_name,
            source=source,
            mime=mime,
            desc_len=len(desc),
        ),
    )
    return ParsedFile(
        modality="image",
        text=desc,
        chunks=[ParsedChunk(text=desc, source=file_name)],
        detail_md=desc,
        meta={"source": source, "file_key": file_key, "url": url, "mime": mime},
    )