from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator

import httpx

from config.config import get_settings

logger = logging.getLogger(__name__)


# ── 通用 LLM 调用函数 ──────────────────────────────────────────────


async def _post_chat_completion(
    *,
    base_url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    stream: bool,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    extra_body: dict[str, Any] | None = None,
) -> httpx.Response:
    """Low-level POST to /v1/chat/completions with timeout."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice
    if extra_body:
        body.update(extra_body)

    url = f"{base_url.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        return await client.post(url, headers=headers, json=body)


async def call_llm(
    *,
    base_url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Non-streaming call to an OpenAI-compatible /v1/chat/completions endpoint.

    Returns the full response dict (OpenAI format).
    """
    t0 = time.perf_counter()
    resp = await _post_chat_completion(
        base_url=base_url,
        model=model,
        api_key=api_key,
        messages=messages,
        stream=False,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    resp.raise_for_status()
    data = resp.json()
    elapsed = time.perf_counter() - t0
    usage = data.get("usage", {})
    logger.info(
        "call_llm model=%s elapsed=%.3fs tokens_in=%s tokens_out=%s",
        model,
        elapsed,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
    )
    return data


async def call_llm_stream(
    *,
    base_url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    extra_body: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Streaming call to an OpenAI-compatible /v1/chat/completions endpoint.

    Yields each chunk dict (OpenAI SSE delta format).
    """
    t0 = time.perf_counter()
    token_count = 0
    resp = await _post_chat_completion(
        base_url=base_url,
        model=model,
        api_key=api_key,
        messages=messages,
        stream=True,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    resp.raise_for_status()
    async for line in resp.aiter_lines():
        if not line or not line.startswith("data: "):
            continue
        data_str = line[len("data: ") :]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        token_count += 1
        yield chunk
    elapsed = time.perf_counter() - t0
    logger.info(
        "call_llm_stream model=%s elapsed=%.3fs tokens=%d",
        model,
        elapsed,
        token_count,
    )


# ── 小模型推理 ────────────────────────────────────────────────────


class SmallModelInference:
    """调用小模型 API（如 Qwen2.5-1.5B）生成快速前缀。

    目标首字响应 < 200ms，使用流式输出。
    """

    def __init__(self) -> None:
        cfg = get_settings().llm.small
        self.base_url = cfg.base_url
        self.model = cfg.model
        self.api_key = cfg.api_key
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature

    async def generate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """非流式调用小模型。"""
        return await call_llm(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    async def generate_stream(
        self, messages: list[dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """流式调用小模型，yield 每个 token 文本。"""
        async for chunk in call_llm_stream(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content


# ── 大模型推理 ────────────────────────────────────────────────────


class LargeModelInference:
    """调用大模型 API（如 qwen3.5-plus）进行深度续写。

    接收小模型的前缀输出作为续写上下文，支持流式输出。
    """

    def __init__(self) -> None:
        cfg = get_settings().llm.large
        self.base_url = cfg.base_url
        self.model = cfg.model
        self.api_key = cfg.api_key
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature

    async def generate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """非流式调用大模型。"""
        return await call_llm(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    async def generate_stream(
        self, messages: list[dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """流式调用大模型，yield 每个 token 文本。"""
        async for chunk in call_llm_stream(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content


# ── 级联流式推理 ──────────────────────────────────────────────────


async def cascaded_stream(
    messages: list[dict[str, Any]],
    *,
    small_model: SmallModelInference | None = None,
    large_model: LargeModelInference | None = None,
) -> AsyncGenerator[str, None]:
    """级联流式推理：小模型快速生成前缀 → 大模型深度续写。

    1. 先从小模型获取前缀（流式 yield 给调用方）。
    2. 收集完整前缀后，作为大模型的输入进行续写（流式 yield 给调用方）。
    3. 实现连贯的流式输出，同时记录耗时和 token 数。

    Args:
        messages: 对话消息列表（OpenAI 格式）。
        small_model: 可选的小模型实例，默认自动创建。
        large_model: 可选的大模型实例，默认自动创建。
    """
    if small_model is None:
        small_model = SmallModelInference()
    if large_model is None:
        large_model = LargeModelInference()

    t_total = time.perf_counter()

    # Phase 1: 小模型快速前缀
    t_small = time.perf_counter()
    small_prefix_parts: list[str] = []
    logger.info("cascaded_stream: starting small model prefix generation")
    async for token in small_model.generate_stream(messages):
        small_prefix_parts.append(token)
        yield token
    t_small_elapsed = time.perf_counter() - t_small
    small_prefix = "".join(small_prefix_parts)
    logger.info(
        "cascaded_stream: small model done elapsed=%.3fs prefix_len=%d",
        t_small_elapsed,
        len(small_prefix),
    )

    if not small_prefix.strip():
        logger.warning("cascaded_stream: small model returned empty prefix, skipping large model")
        return

    # Phase 2: 大模型深度续写
    # 将小模型输出作为 assistant 消息追加到 messages，让大模型续写
    continuation_messages = [
        *messages,
        {"role": "assistant", "content": small_prefix},
    ]

    t_large = time.perf_counter()
    large_token_count = 0
    logger.info("cascaded_stream: starting large model continuation")
    async for token in large_model.generate_stream(continuation_messages):
        large_token_count += 1
        yield token
    t_large_elapsed = time.perf_counter() - t_large

    t_total_elapsed = time.perf_counter() - t_total
    logger.info(
        "cascaded_stream: done total=%.3fs small=%.3fs large=%.3fs "
        "small_chars=%d large_tokens=%d",
        t_total_elapsed,
        t_small_elapsed,
        t_large_elapsed,
        len(small_prefix),
        large_token_count,
    )