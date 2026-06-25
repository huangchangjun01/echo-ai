from __future__ import annotations

import json
import logging
from typing import Any

from config.config import get_settings
from llm.inference import call_llm

logger = logging.getLogger(__name__)

_EMOTION_PROMPT = """你是一个情感分析助手。分析以下文本的情感，返回严格的 JSON 格式。

文本：{text}

请返回如下 JSON（不要包含其他内容）：
{{"emotion": "<joy/sadness/anger/fear/surprise/disgust/neutral>", "intensity": <0.0-1.0>, "confidence": <0.0-1.0>}}"""


async def analyze_emotion(text: str) -> dict[str, Any]:
    """使用小模型快速分析文本情感。

    Args:
        text: 待分析的文本。

    Returns:
        {"emotion": "...", "intensity": 0.0-1.0, "confidence": 0.0-1.0}
    """
    cfg = get_settings().llm.small
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _EMOTION_PROMPT.format(text=text)},
    ]

    try:
        resp = await call_llm(
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.api_key,
            messages=messages,
            temperature=0.3,
            max_tokens=128,
        )
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        content = content.strip()

        # 尝试提取 JSON
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        result = json.loads(content)
        return {
            "emotion": result.get("emotion", "neutral"),
            "intensity": float(result.get("intensity", 0.0)),
            "confidence": float(result.get("confidence", 0.0)),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Failed to parse emotion result: %s, raw=%s", e, content)
        return {"emotion": "neutral", "intensity": 0.0, "confidence": 0.0}
    except Exception as e:
        logger.error("analyze_emotion failed: %s", e)
        return {"emotion": "neutral", "intensity": 0.0, "confidence": 0.0}


def get_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "analyze_emotion",
            "description": "分析文本情感，返回情感类型、强度和置信度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "待分析情感的文本",
                    },
                },
                "required": ["text"],
            },
        },
    }