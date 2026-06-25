from __future__ import annotations

import base64
import logging
from typing import Any

from config.config import get_settings
from llm.inference import call_llm

logger = logging.getLogger(__name__)


async def understand_image(image_data: bytes, question: str = "请描述这张图片的内容") -> str:
    """使用大模型 vision 能力理解图片内容。

    Args:
        image_data: 图片原始字节数据。
        question: 对图片的提问，默认询问图片内容描述。

    Returns:
        图片描述文本。
    """
    cfg = get_settings().llm.large
    encoded = base64.b64encode(image_data).decode("utf-8")

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                {"type": "text", "text": question},
            ],
        }
    ]

    try:
        resp = await call_llm(
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.api_key,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip()
    except Exception as e:
        logger.error("understand_image failed: %s", e)
        return f"图片理解失败: {e}"


def get_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "understand_image",
            "description": "使用视觉模型理解图片内容，返回图片描述。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_data": {
                        "type": "string",
                        "description": "图片的 base64 编码数据",
                    },
                    "question": {
                        "type": "string",
                        "description": "对图片的提问，默认为'请描述这张图片的内容'",
                    },
                },
                "required": ["image_data"],
            },
        },
    }