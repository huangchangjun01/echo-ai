from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

from config.config import get_settings
from llm.inference import call_llm, call_llm_stream

logger = logging.getLogger(__name__)

# ── 工具注册表 ────────────────────────────────────────────────────

_TOOL_REGISTRY: dict[str, Any] = {}


def register_tool(name: str, fn: Any) -> None:
    """注册工具执行函数，供 execute_tool_call 路由使用。"""
    _TOOL_REGISTRY[name] = fn


def build_tool_definitions() -> list[dict[str, Any]]:
    """从工具模块构建 OpenAI 兼容的 tools 定义。

    委托给 utils.tools.tool_schemas() 以保持单一来源。
    """
    from utils.tools import tool_schemas as _tool_schemas

    return _tool_schemas()


async def execute_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    """根据 tool_name 路由到对应的工具执行函数，返回结果字符串。

    Args:
        tool_name: 工具名称（如 "vector_search"）。
        arguments: 工具参数（已解析的 dict）。

    Returns:
        工具执行结果（JSON 字符串）。
    """
    fn = _TOOL_REGISTRY.get(tool_name)
    if fn is None:
        msg = f"Unknown tool: {tool_name}"
        logger.warning("execute_tool_call: %s", msg)
        return json.dumps({"error": msg})

    logger.info("execute_tool_call: %s args=%s", tool_name, arguments)
    import asyncio

    if asyncio.iscoroutinefunction(fn):
        result = await fn(**arguments)
    else:
        result = await asyncio.to_thread(fn, **arguments)

    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


# ── 系统提示词构建 ────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """你是一个友好、乐于助人的 AI 助手，名叫 Echo。

## 人格设定
- 你热情、耐心、善于倾听。
- 你回答简洁清晰，避免冗长。
- 你会在适当的时候使用工具来获取信息。

## L0 核心记忆
{memory_context}

## 工具使用指南
- 当需要检索用户文档或知识库时，使用 vector_search 工具。
- 工具参数中的 user_id 必须使用当前用户 ID。
- 获取工具结果后，基于结果给出自然语言回答。
"""


def _build_system_prompt(memory_context: str = "") -> str:
    """构建系统提示词，包含人格设定和 L0 核心记忆。"""
    ctx = memory_context.strip() if memory_context.strip() else "暂无核心记忆。"
    return _SYSTEM_PROMPT_TEMPLATE.format(memory_context=ctx)


# ── ReAct 循环 ────────────────────────────────────────────────────


class ReActLoop:
    """轻量 ReAct 循环：推理 → 行动 → 观察 → 推理 ...

    循环执行：
    1. 调用 LLM（带 tools 定义）
    2. 检查是否有 tool_calls
    3. 执行工具调用
    4. 将工具结果反馈给 LLM
    5. 重复直到无 tool_calls 或达到最大循环次数

    Args:
        messages: 对话消息列表（不含 system 消息）。
        tools: OpenAI 兼容的 tools 定义列表。
        memory_context: L0 核心记忆文本（可选）。
        max_iterations: 最大循环次数，默认 5。
        model_config: 模型配置 dict，包含 base_url, model, api_key 等。
                      默认使用大模型配置。
    """

    def __init__(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        memory_context: str = "",
        max_iterations: int = 5,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self._messages = messages
        self._tools = tools or build_tool_definitions()
        self._memory_context = memory_context
        self._max_iterations = max_iterations
        if model_config is None:
            cfg = get_settings().llm.large
            model_config = {
                "base_url": cfg.base_url,
                "model": cfg.model,
                "api_key": cfg.api_key,
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
            }
        self._model_config = model_config

    def _build_full_messages(self) -> list[dict[str, Any]]:
        """构建包含系统提示词的完整消息列表。"""
        system_prompt = _build_system_prompt(self._memory_context)
        return [
            {"role": "system", "content": system_prompt},
            *self._messages,
        ]

    async def run(self) -> str:
        """执行 ReAct 循环，返回最终回复文本。"""
        messages = self._build_full_messages()
        iteration = 0

        while iteration < self._max_iterations:
            iteration += 1
            logger.info("ReAct iteration %d/%d", iteration, self._max_iterations)

            resp = await call_llm(
                messages=messages,
                tools=self._tools,
                **self._model_config,
            )

            choice = resp.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                # 无工具调用，返回最终回复
                return message.get("content", "")

            # 将 assistant 消息（含 tool_calls）加入上下文
            messages.append(message)

            # 执行每个工具调用
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                try:
                    arguments = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}

                tool_result = await execute_tool_call(tool_name, arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_result,
                })

        logger.warning("ReAct loop reached max iterations (%d)", self._max_iterations)

        # 达到最大循环次数，做最后一次不带工具的回答
        resp = await call_llm(
            messages=messages,
            tools=None,
            **self._model_config,
        )
        choice = resp.get("choices", [{}])[0]
        return choice.get("message", {}).get("content", "")

    async def run_stream(self) -> AsyncGenerator[str, None]:
        """流式执行 ReAct 循环，在最终回复阶段流式 yield token。

        中间的推理和工具调用阶段不流式输出（仅最终回复流式返回）。
        """
        messages = self._build_full_messages()
        iteration = 0

        while iteration < self._max_iterations:
            iteration += 1
            logger.info("ReAct stream iteration %d/%d", iteration, self._max_iterations)

            resp = await call_llm(
                messages=messages,
                tools=self._tools,
                **self._model_config,
            )

            choice = resp.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                # 无工具调用，进入流式最终回复阶段
                messages.append(message)
                async for chunk in call_llm_stream(
                    messages=messages,
                    tools=None,
                    **self._model_config,
                ):
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                return

            # 将 assistant 消息（含 tool_calls）加入上下文
            messages.append(message)

            # 执行每个工具调用
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                try:
                    arguments = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}

                tool_result = await execute_tool_call(tool_name, arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_result,
                })

        logger.warning("ReAct stream loop reached max iterations (%d)", self._max_iterations)

        # 达到最大循环次数，流式输出最终回复
        async for chunk in call_llm_stream(
            messages=messages,
            tools=None,
            **self._model_config,
        ):
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content