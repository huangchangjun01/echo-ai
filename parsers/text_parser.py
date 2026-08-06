"""文本解析器：纯文本/代码/markdown 直接下载后按行处理。

二进制内容安全护栏：
    拿到 raw 字节后，先做"是否可读文本"的快速判定；若明显是二进制（如 mp4/jpg/zip），
    直接判失败，绝不把乱码写进 detail_md（下游会被 LLM 当正文总结）。
    修复视频编辑重解析 bug 的最后一道防线。
"""

from __future__ import annotations

import logging
import re
import unicodedata

from utils.downloader import fetch_source_bytes
from utils.request_context import log_exception
from .base import ParsedFile, ParsedChunk

logger = logging.getLogger(__name__)

# 控制字符阈值：文本里出现 ≥ _BINARY_CTRL_THRESHOLD 个比例的控制字符（含 \x00）
# 就视为二进制。0.5% 在 mp4/jpg 上稳定命中，正常中文文本几乎为 0。
_BINARY_CTRL_THRESHOLD = 0.005
_BINARY_MIN_BYTES = 64  # 小于这个字节数不做判断（极短文本可能凑巧命中）

# 常见二进制 magic（前 8 字节）。命中其一即视为二进制。
_BINARY_MAGIC: tuple[bytes, ...] = (
    b"\x00\x00\x00",  # mp4 / mov / m4v 的 box size 前缀
    b"\xff\xd8\xff",  # JPEG
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"GIF87a",  # GIF
    b"GIF89a",  # GIF
    b"PK\x03\x04",  # ZIP / DOCX / XLSX / JAR
    b"%PDF-",  # PDF
    b"7z\xbc\xaf\x27\x1c",  # 7z
    b"Rar!\x1a\x07",  # RAR
    b"\x1f\x8b",  # gzip
    b"\x42\x4d",  # BMP
    b"RIFF",  # WAV / AVI / WEBP（容器起始）
)

# 解码后兜底：解码文本里若出现典型二进制盒/编码字段字符串，直接判失败
# （这些 ASCII 串几乎不可能出现在合法文本里，但 decode('utf-8','replace') 后会出现）
_BINARY_TEXT_SIGNALS: tuple[str, ...] = (
    "ftypisom", "ftypmp42", "ftypmp41", "ftypqt  ", "moov", "mvhd", "tkhd", "mdat",
    "stts", "stsc", "stsz", "stco", "esds", "avc1", "avcC", "hdlr",
    "Exif", "IHDR", "IDAT", "IEND",
)


def _looks_binary(raw: bytes) -> tuple[bool, str]:
    """返回 (是否二进制, 原因)。"""
    if len(raw) < _BINARY_MIN_BYTES:
        return False, "too_short"

    # 1) magic 头
    head = raw[:8]
    for sig in _BINARY_MAGIC:
        if head.startswith(sig):
            return True, f"magic:{sig[:4]!r}"

    # 2) 控制字符 / NUL 比例
    if b"\x00" in raw:
        n_null = raw.count(b"\x00")
        # NUL 出现一次（0x00 字节）是 BMO/UTF-16 BOM 等边角，但密级 0.5% 以上基本就是二进制
        if n_null / len(raw) >= 0.02:
            return True, f"nul_ratio={n_null / len(raw):.3f}"

    # 控制字符（C0/C1，剔除常见换行/制表）
    bad = sum(1 for b in raw if b < 0x20 and b not in (0x09, 0x0A, 0x0D))
    if bad / len(raw) >= _BINARY_CTRL_THRESHOLD:
        return True, f"ctrl_ratio={bad / len(raw):.3f}"

    # 3) 解码后特征串（兜底覆盖"扩展名被错改成 .txt"的 mp4 场景）
    try:
        decoded = raw[:4096].decode("utf-8", errors="replace")
    except Exception:
        return True, "decode_failed"
    for sig in _BINARY_TEXT_SIGNALS:
        if sig in decoded:
            return True, f"text_signal:{sig}"

    return False, "ok"


async def parse_text(file_key: str, file_name: str, url: str | None) -> ParsedFile:
    if not file_key and not url:
        return ParsedFile(modality="text", text="", meta={"error": "no_key_or_url"})
    try:
        # fileKey 优先直连七牛，url 仅兜底：避免把 SPA fallback / 错误页 HTML 当正文
        raw = await fetch_source_bytes(file_key or None, url or None)

        # 二进制护栏：明确是二进制的源（mp4/jpg/zip/...）绝不能当文本喂给 LLM
        is_bin, reason = _looks_binary(raw)
        if is_bin:
            log_exception(
                logger,
                "parse_text 拒绝二进制内容",
                exc=None,
                level=logging.WARNING,
                stage="parser_text",
                event="binary_rejected",
                file_name=file_name,
                reason=reason,
                bytes=len(raw),
            )
            return ParsedFile(
                modality="text",
                text="",
                meta={
                    "error": f"binary_rejected:{reason}",
                    "hint": "前端 fileType 与文件实际类型不一致，已拒绝按文本解析",
                },
            )

        # 简单去 BOM 与解码失败兜底
        text = raw.decode("utf-8", errors="replace")
        text = text.lstrip("﻿").strip()
        # 防御：标准化后若全是非打印字符（极少见），也视为失败
        if text and not any(unicodedata.category(c).startswith(("L", "N", "P", "Z")) for c in text[:512]):
            return ParsedFile(
                modality="text",
                text="",
                meta={"error": "no_printable_text"},
            )
        # 把异常长的连续空白折叠，避免把"半截 think 块"之类串进 detail_md
        text = re.sub(r"[ \t]{20,}", "  ", text)
        return ParsedFile(
            modality="text",
            text=text[:2000],  # 主文本截断
            chunks=[ParsedChunk(text=text, source=file_name)],
            detail_md=text,
            meta={"bytes": len(raw)},
        )
    except Exception as e:
        log_exception(
            logger,
            "parse_text failed",
            exc=e,
            level=logging.WARNING,
            stage="parser_text",
            event="error",
            file_name=file_name,
        )
        return ParsedFile(modality="text", text="", meta={"error": str(e)[:200]})