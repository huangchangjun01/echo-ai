"""解析器注册表：按 fileType 与 MIME 分派到具体实现。"""

from __future__ import annotations

from .base import ParsedFile
from .text_parser import parse_text
from .image_parser import parse_image
from .audio_parser import parse_audio
from .video_parser import parse_video


async def parse_file(file_key: str, file_name: str, file_type: int, file_url: str | None) -> ParsedFile:
    """按 fileType 分派；fileType 与 file 表约定一致：1=文本 2=图片 3=视频 4=音频。"""
    if file_type == 1:
        return await parse_text(file_key, file_name, file_url)
    if file_type == 2:
        return await parse_image(file_key, file_name, file_url)
    if file_type == 3:
        return await parse_video(file_key, file_name, file_url)
    if file_type == 4:
        return await parse_audio(file_key, file_name, file_url)
    raise ValueError(f"unknown fileType={file_type}")