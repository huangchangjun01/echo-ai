from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── 全局工具注册表 ─────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Any] = {}
_TOOL_DEFINITIONS: dict[str, Any] = {}


def register_tool(name: str, fn: Any, definition_fn: Any = None) -> None:
    """注册工具到全局注册表。"""
    if name in TOOL_REGISTRY:
        logger.warning("Tool %s already registered, overwriting.", name)
    TOOL_REGISTRY[name] = fn
    if definition_fn is not None:
        _TOOL_DEFINITIONS[name] = definition_fn
    logger.info("Registered tool: %s", name)


def get_tool(name: str) -> Any | None:
    """按名称获取工具。"""
    return TOOL_REGISTRY.get(name)


def get_all_tools() -> dict[str, Any]:
    """获取所有已注册工具。"""
    return dict(TOOL_REGISTRY)


def build_tool_definitions() -> list[dict[str, Any]]:
    """构建 OpenAI 兼容的 tools 定义列表。"""
    definitions: list[dict[str, Any]] = []
    for name in TOOL_REGISTRY:
        def_fn = _TOOL_DEFINITIONS.get(name)
        if def_fn is not None:
            definitions.append(def_fn())
    return definitions


# ── 自动注册内置工具 ───────────────────────────────────────────────

from tools.analyze_emotion import analyze_emotion, get_tool_definition as _analyze_emotion_def  # noqa: E402
from tools.search_memory import search_memory, get_tool_definition as _search_memory_def  # noqa: E402
from tools.understand_audio import understand_audio, get_tool_definition as _understand_audio_def  # noqa: E402
from tools.understand_image import understand_image, get_tool_definition as _understand_image_def  # noqa: E402

register_tool("analyze_emotion", analyze_emotion, _analyze_emotion_def)
register_tool("search_memory", search_memory, _search_memory_def)
register_tool("understand_audio", understand_audio, _understand_audio_def)
register_tool("understand_image", understand_image, _understand_image_def)