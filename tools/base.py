"""工具基类：所有工具统一暴露 `name/description/schema/run` 接口。

- 进程内直接调用（按用户要求「进程内」执行视觉/语音/检索/情感）。
- 每个工具都允许返回纯 dict，便于 ReAct 循环直接塞回 LLM。
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolResult:
    """统一工具返回。`ok=True` 表示成功，data 是结果 dict。"""

    __slots__ = ("ok", "data", "error")

    def __init__(self, ok: bool, data: Any = None, error: str | None = None) -> None:
        self.ok = ok
        self.data = data
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "data": self.data, "error": self.error}


class BaseTool:
    """进程内工具基类。"""

    name: str = "base"
    description: str = ""

    def run(self, **kwargs: Any) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        """OpenAI 风格 function schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._args_schema() if hasattr(self, "_args_schema") else {"type": "object"},
            },
        }


def tool_input(**fields: Any) -> type[BaseModel]:
    """快速声明工具入参的 pydantic 模型。"""

    return type(
        "ToolInput",
        (BaseModel,),
        {k: v for k, v in fields.items()},
    )  # type: ignore[return-value]


def ok(data: Any) -> ToolResult:
    return ToolResult(ok=True, data=data)


def fail(error: str) -> ToolResult:
    return ToolResult(ok=False, error=error)