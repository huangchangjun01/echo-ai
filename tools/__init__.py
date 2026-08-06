"""tools 包：5 个进程内工具的统一注册与执行。"""

import json
import logging
import time

from .analyze_emotion import AnalyzeEmotionTool
from .base import BaseTool, ToolResult, fail, ok
from .read_memory_full import ReadMemoryFullTool
from .search_memory import SearchMemoryTool
from .understand_audio import UnderstandAudioTool
from .understand_image import UnderstandImageTool
from utils.request_context import log_exception, merge_extra

logger = logging.getLogger(__name__)

__all__ = [
    "AnalyzeEmotionTool",
    "BaseTool",
    "ReadMemoryFullTool",
    "SearchMemoryTool",
    "ToolResult",
    "UnderstandAudioTool",
    "UnderstandImageTool",
    "fail",
    "ok",
]


# ---------- 工具注册表 ----------

_REGISTRY: dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> None:
    _REGISTRY[tool.name] = tool


def get_tool(name: str) -> BaseTool | None:
    return _REGISTRY.get(name)


def list_tools() -> list[BaseTool]:
    return list(_REGISTRY.values())


def tool_schemas() -> list[dict]:
    return [t.schema() for t in list_tools()]


def dispatch(name: str, **kwargs) -> ToolResult:
    tool = _REGISTRY.get(name)
    if tool is None:
        return fail(f"unknown tool: {name}")
    import asyncio

    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        loop_running = False
    t0 = time.perf_counter()
    logger.info(
        "tool.dispatch start",
        extra=merge_extra(
            stage=f"tool_{name}",
            event="start",
            args_summary=json.dumps(kwargs, ensure_ascii=False, default=str)[:200],
        ),
    )
    try:
        if loop_running and hasattr(tool, "arun"):
            return fail("called dispatch() in event loop; use adispatch() instead")
        result = tool.run(**kwargs)
        logger.info(
            "tool.dispatch ok",
            extra=merge_extra(
                stage=f"tool_{name}",
                event="end",
                ok=bool(result.ok),
                err=(result.error or "")[:160] if not result.ok else "",
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return result
    except Exception as e:
        log_exception(
            logger,
            "tool.dispatch failed",
            exc=e,
            level=logging.ERROR,
            stage=f"tool_{name}",
            event="error",
            args_summary=json.dumps(kwargs, ensure_ascii=False, default=str)[:200],
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return fail(str(e))


async def adispatch(name: str, **kwargs) -> ToolResult:
    """异步版 dispatch：始终走 async 路径，避免事件循环嵌套。"""
    tool = _REGISTRY.get(name)
    if tool is None:
        return fail(f"unknown tool: {name}")
    t0 = time.perf_counter()
    logger.info(
        "tool.adispatch start",
        extra=merge_extra(
            stage=f"tool_{name}",
            event="start",
            args_summary=json.dumps(kwargs, ensure_ascii=False, default=str)[:200],
        ),
    )
    try:
        if hasattr(tool, "arun"):
            result = await tool.arun(**kwargs)  # type: ignore[attr-defined]
        else:
            import asyncio

            result = await asyncio.to_thread(tool.run, **kwargs)
        logger.info(
            "tool.adispatch ok",
            extra=merge_extra(
                stage=f"tool_{name}",
                event="end",
                ok=bool(result.ok),
                err=(result.error or "")[:160] if not result.ok else "",
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return result
    except Exception as e:
        log_exception(
            logger,
            "tool.adispatch failed",
            exc=e,
            level=logging.ERROR,
            stage=f"tool_{name}",
            event="error",
            args_summary=json.dumps(kwargs, ensure_ascii=False, default=str)[:200],
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return fail(str(e))


# ---------- 启动时注册默认 5 个工具 ----------

register_tool(UnderstandImageTool())
register_tool(UnderstandAudioTool())
register_tool(SearchMemoryTool())
register_tool(ReadMemoryFullTool())
register_tool(AnalyzeEmotionTool())