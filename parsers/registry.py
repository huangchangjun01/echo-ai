"""解析器注册表：按 fileType 与扩展名分派到具体实现。

权威顺序：扩展名 > 声明的 fileType。
当上游把视频文件误标成 1(文本) 时，扩展名兜底可保证不会把 mp4 字节流
decode 成文本喂给 LLM（用户反馈的"片段里出现 ftyp/moov/avc1/mp4a 元数据"bug）。
"""

from __future__ import annotations

import logging
import os

from utils.request_context import log_exception, merge_extra
from .base import ParsedFile
from .text_parser import parse_text
from .image_parser import parse_image
from .audio_parser import parse_audio
from .video_parser import parse_video

logger = logging.getLogger("echo-ai.parsers")

# 扩展名 → fileType。键必须小写（lookup 时走 .lower()）。
_EXT_TO_FILE_TYPE: dict[str, int] = {
    # 视频
    "mp4": 3, "mov": 3, "m4v": 3, "mkv": 3, "webm": 3, "avi": 3, "flv": 3, "wmv": 3, "3gp": 3,
    # 图片
    "jpg": 2, "jpeg": 2, "png": 2, "gif": 2, "webp": 2, "bmp": 2, "heic": 2, "heif": 2, "tiff": 2, "tif": 2,
    # 音频
    "mp3": 4, "wav": 4, "m4a": 4, "aac": 4, "flac": 4, "ogg": 4, "opus": 4, "wma": 4, "amr": 4,
    # 文本 / 文档
    "txt": 1, "md": 1, "markdown": 1, "log": 1, "csv": 1, "json": 1, "xml": 1, "html": 1, "htm": 1, "yaml": 1, "yml": 1,
}


def _file_type_by_ext(file_name: str) -> int:
    """按扩展名推断 fileType；无法识别返回 0。"""
    if not file_name:
        return 0
    _, ext = os.path.splitext(file_name)
    if not ext:
        return 0
    return _EXT_TO_FILE_TYPE.get(ext.lower().lstrip("."), 0)


def _resolve_file_type(file_name: str, declared: int) -> tuple[int, str]:
    """权威 fileType 解析：扩展名 > 声明值；冲突时记告警。"""
    ext_type = _file_type_by_ext(file_name)
    if ext_type and ext_type != declared:
        if declared not in (0, 1, 2, 3, 4):
            return ext_type, "ext_overrides_invalid_declared"
        # 1 是个特例：未知扩展名/无扩展名时请求方常默认传 1，不算冲突
        if declared == 1 and ext_type != 0:
            return ext_type, "ext_overrides_text_default"
        return ext_type, "ext_overrides_declared"
    return declared, "ok" if ext_type == 0 or ext_type == declared else "declared_wins"


async def parse_file(file_key: str, file_name: str, file_type: int, file_url: str | None) -> ParsedFile:
    """按 fileType 分派；fileType 与 file 表约定一致：1=文本 2=图片 3=视频 4=音频。

    冲突检测：扩展名能识别时，扩展名权威，避免"mp4 被前端误标成文本→原文塞进 prompt"。
    """
    resolved, reason = _resolve_file_type(file_name or "", file_type)
    if reason != "ok":
        try:
            logger.warning(
                "registry: fileType 冲突，已按扩展名纠正",
                extra=merge_extra(
                    stage="parser_registry",
                    event="fileType_override",
                    file_name=file_name,
                    declared=file_type,
                    resolved=resolved,
                    reason=reason,
                ),
            )
        except Exception:  # noqa: BLE001
            pass  # 上下文日志不可用时不影响主流程
    file_type = resolved

    if file_type == 1:
        return await parse_text(file_key, file_name, file_url)
    if file_type == 2:
        return await parse_image(file_key, file_name, file_url)
    if file_type == 3:
        return await parse_video(file_key, file_name, file_url)
    if file_type == 4:
        return await parse_audio(file_key, file_name, file_url)
    raise ValueError(f"unknown fileType={file_type}")