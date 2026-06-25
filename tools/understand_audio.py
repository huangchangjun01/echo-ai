from __future__ import annotations

import logging
from typing import Any

from embedding.whisper import transcribe

logger = logging.getLogger(__name__)


def understand_audio(audio_data: bytes) -> str:
    """使用 Whisper 将音频转录为文本。

    Args:
        audio_data: 音频原始字节数据。

    Returns:
        转录文本，失败时返回错误信息。
    """
    try:
        text = transcribe(audio_data)
        return text
    except Exception as e:
        logger.error("understand_audio failed: %s", e)
        return f"音频转录失败: {e}"


def get_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "understand_audio",
            "description": "使用 Whisper 模型将音频转录为文本。",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_data": {
                        "type": "string",
                        "description": "音频的 base64 编码数据",
                    },
                },
                "required": ["audio_data"],
            },
        },
    }