"""LLM 客户端：OpenAI 兼容协议。

- 小模型：负责**日常对话**（方案 A：单一小模型直接流式生成回复，首字延迟敏感），
  同时承担意图分类、ReAct 工具决策等对话链路内的生成。
- 大模型：负责**记忆抽取 / 摘要生成 / 图文音视频内容描述**等后台生成任务。
- 两个端点可独立配置（`LLM_SMALL_BASE_URL` / `LLM_SMALL_API_KEY`），缺省时复用大模型配置。
- 解析失败时**返回原始文本**而不是抛异常，方便上层 ReAct 循环对错误兜底。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from config.config import get_settings
from utils.request_context import log_exception, log_silent_failure, merge_extra

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI 兼容 LLM 客户端（同步+异步 + 流式）。"""

    def __init__(self) -> None:
        cfg = get_settings().llm
        self.model = cfg.model
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
        self._client = AsyncOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=60.0,
        )

        # 小模型配置（情感微模型 / 前缀生成）
        small_base, small_key = cfg.small_resolved()
        if small_base and small_key:
            self._small_client = AsyncOpenAI(
                api_key=small_key,
                base_url=small_base,
                timeout=30.0,
            )
        else:
            self._small_client = self._client
        self.small_model = cfg.small_model
        self.small_max_tokens = cfg.small_max_tokens
        self.small_temperature = cfg.small_temperature

    # ---------- 大模型 ----------

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """非流式 chat completion。返回 OpenAI 原始 dict。"""
        eff_temp = temperature if temperature is not None else self.temperature
        eff_max = max_tokens if max_tokens is not None else self.max_tokens
        t0 = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=eff_temp,
                max_tokens=eff_max,
                tools=tools,  # type: ignore[arg-type]
            )
        except Exception as e:
            log_exception(
                logger,
                "LLM chat failed",
                exc=e,
                level=logging.ERROR,
                include_traceback=True,
                stage="llm_chat",
                event="error",
                model=self.model,
                msg_count=len(messages),
                max_tokens=eff_max,
                temperature=eff_temp,
                has_tools=bool(tools),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
            raise
        data = _to_dict(resp)
        if _should_retry_starved(data, eff_max):
            # DeepSeek 推理思考吃光 max_tokens 导致 content 为空：翻倍预算重试一次，
            # 让工具调用 JSON / 结构化输出能够落地。
            retry_max = max(eff_max * 2, 1024)
            logger.warning(
                "LLM chat retry (reasoning starved budget)",
                extra=merge_extra(
                    stage="llm_chat",
                    event="retry_starved",
                    model=self.model,
                    msg_count=len(messages),
                    eff_max=eff_max,
                    retry_max=retry_max,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                ),
            )
            try:
                resp2 = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=eff_temp,
                    max_tokens=retry_max,
                    tools=tools,  # type: ignore[arg-type]
                )
                data = _to_dict(resp2)
            except Exception as e:
                log_exception(
                    logger,
                    "LLM chat retry failed (use first result)",
                    exc=e,
                    level=logging.WARNING,
                    include_traceback=False,
                    stage="llm_chat",
                    event="retry_error",
                    model=self.model,
                )
        try:
            usage = data.get("usage") or {}
            logger.info(
                "LLM chat ok",
                extra=merge_extra(
                    stage="llm_chat",
                    event="ok",
                    model=self.model,
                    msg_count=len(messages),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    choices=len(data.get("choices", []) or []),
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                ),
            )
        except Exception as e:
            log_silent_failure(
                logger,
                "usage log on chat ok failed (skipped)",
                exc=e,
                stage="llm_chat",
                event="usage_log_error",
                model=self.model,
            )
        return data

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """大模型流式输出，逐 chunk yield ``(正文增量, 思考增量)``。

        DeepSeek 风格模型把推理过程放在独立的 ``reasoning_content`` 字段
        （``delta.reasoning_content``），与正文 ``content`` 分离；两者任一非空即
        yield，调用方分别消费（思考下发为 thinking 事件，正文下发为 delta）。
        """
        eff_temp = temperature if temperature is not None else self.temperature
        eff_max = max_tokens if max_tokens is not None else self.max_tokens
        t0 = time.perf_counter()
        first_delta_at: float | None = None
        delta_count = 0
        out_chars = 0
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=eff_temp,
                max_tokens=eff_max,
                stream=True,
            )
            async for chunk in stream:
                try:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    content = choice.delta.content or ""
                    reasoning = getattr(choice.delta, "reasoning_content", None) or ""
                    if not content and not reasoning:
                        continue
                    if content:
                        if first_delta_at is None:
                            first_delta_at = time.perf_counter()
                        delta_count += 1
                        out_chars += len(content)
                    yield content, reasoning
                except Exception as e:
                    log_silent_failure(
                        logger,
                        "stream chunk decode failed (skip)",
                        exc=e,
                        stage="llm_stream",
                        event="chunk_decode_error",
                        model=self.model,
                    )
                    continue
        except Exception as e:
            log_exception(
                logger,
                "LLM stream failed",
                exc=e,
                level=logging.ERROR,
                include_traceback=True,
                stage="llm_stream",
                event="error",
                model=self.model,
                msg_count=len(messages),
                max_tokens=eff_max,
                temperature=eff_temp,
                delta_count=delta_count,
                out_chars=out_chars,
                first_delta_ms=round((first_delta_at - t0) * 1000, 2) if first_delta_at else None,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
            raise
        finally:
            try:
                logger.info(
                    "LLM stream end",
                    extra=merge_extra(
                        stage="llm_stream",
                        event="end",
                        model=self.model,
                        msg_count=len(messages),
                        delta_count=delta_count,
                        out_chars=out_chars,
                        reasoning_chars=0,
                        first_delta_ms=round((first_delta_at - t0) * 1000, 2) if first_delta_at else None,
                        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    ),
                )
            except Exception as e:
                log_silent_failure(
                    logger,
                    "stream end log failed (skipped)",
                    exc=e,
                    stage="llm_stream",
                    event="end_log_error",
                    model=self.model,
                )

    # ---------- 小模型（日常对话链路） ----------

    async def small_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """小模型非流式 chat（对话链路内：ReAct 工具决策等）。

        与大模型 ``chat`` 同构，但走 ``_small_client``（小模型端点）。
        """
        eff_temp = temperature if temperature is not None else self.small_temperature
        eff_max = max_tokens if max_tokens is not None else self.small_max_tokens
        t0 = time.perf_counter()
        try:
            resp = await self._small_client.chat.completions.create(
                model=self.small_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=eff_temp,
                max_tokens=eff_max,
                tools=tools,  # type: ignore[arg-type]
            )
        except Exception as e:
            log_exception(
                logger,
                "small LLM chat failed",
                exc=e,
                level=logging.WARNING,
                include_traceback=True,
                stage="small_chat",
                event="error",
                model=self.small_model,
                msg_count=len(messages),
                max_tokens=eff_max,
                temperature=eff_temp,
                has_tools=bool(tools),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
            raise
        data = _to_dict(resp)
        if _should_retry_starved(data, eff_max):
            # DeepSeek 推理思考吃光 max_tokens 导致 content 为空：翻倍预算重试一次，
            # 让 ReAct 决策的工具调用 JSON 能够落地。
            retry_max = max(eff_max * 2, 1024)
            logger.warning(
                "small LLM chat retry (reasoning starved budget)",
                extra=merge_extra(
                    stage="small_chat",
                    event="retry_starved",
                    model=self.small_model,
                    msg_count=len(messages),
                    eff_max=eff_max,
                    retry_max=retry_max,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                ),
            )
            try:
                resp2 = await self._small_client.chat.completions.create(
                    model=self.small_model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=eff_temp,
                    max_tokens=retry_max,
                    tools=tools,  # type: ignore[arg-type]
                )
                data = _to_dict(resp2)
            except Exception as e:
                log_exception(
                    logger,
                    "small LLM chat retry failed (use first result)",
                    exc=e,
                    level=logging.WARNING,
                    include_traceback=False,
                    stage="small_chat",
                    event="retry_error",
                    model=self.small_model,
                )
        try:
            usage = data.get("usage") or {}
            logger.info(
                "small LLM chat ok",
                extra=merge_extra(
                    stage="small_chat",
                    event="ok",
                    model=self.small_model,
                    msg_count=len(messages),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    choices=len(data.get("choices", []) or []),
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                ),
            )
        except Exception as e:
            log_silent_failure(
                logger,
                "usage log on small chat ok failed (skipped)",
                exc=e,
                stage="small_chat",
                event="usage_log_error",
                model=self.small_model,
            )
        return data

    async def small_prefix(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """小模型单次生成文本（意图分类等短输出场景；失败返回空串，不阻断流程）。"""
        try:
            data = await self.small_chat(messages, max_tokens=max_tokens, temperature=temperature)
            return _extract_text(data) or ""
        except Exception:
            return ""

    async def small_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """小模型流式输出（日常对话正文生成：单一小模型直接流式）。

        逐 chunk yield ``(正文增量, 思考增量)``；DeepSeek 风格模型会把推理过程
        放在独立的 ``reasoning_content`` 字段，调用方将思考下发为 thinking 事件、
        正文下发为 delta。
        """
        eff_temp = temperature if temperature is not None else self.small_temperature
        eff_max = max_tokens if max_tokens is not None else self.small_max_tokens
        t0 = time.perf_counter()
        first_delta_at: float | None = None
        delta_count = 0
        out_chars = 0
        try:
            stream = await self._small_client.chat.completions.create(
                model=self.small_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=eff_temp,
                max_tokens=eff_max,
                stream=True,
            )
            async for chunk in stream:
                try:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    content = choice.delta.content or ""
                    reasoning = getattr(choice.delta, "reasoning_content", None) or ""
                    if not content and not reasoning:
                        continue
                    if content:
                        if first_delta_at is None:
                            first_delta_at = time.perf_counter()
                        delta_count += 1
                        out_chars += len(content)
                    yield content, reasoning
                except Exception as e:
                    log_silent_failure(
                        logger,
                        "small_stream chunk decode failed (skip)",
                        exc=e,
                        stage="small_stream",
                        event="chunk_decode_error",
                        model=self.small_model,
                    )
                    continue
        except Exception as e:
            log_exception(
                logger,
                "small_stream failed",
                exc=e,
                level=logging.WARNING,
                stage="small_stream",
                event="error",
                model=self.small_model,
                msg_count=len(messages),
                temperature=eff_temp,
                max_tokens=eff_max,
                delta_count=delta_count,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
            return
        finally:
            try:
                logger.info(
                    "small_stream end",
                    extra=merge_extra(
                        stage="small_stream",
                        event="end",
                        model=self.small_model,
                        delta_count=delta_count,
                        out_chars=out_chars,
                        first_delta_ms=round((first_delta_at - t0) * 1000, 2) if first_delta_at else None,
                        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    ),
                )
            except Exception as e:
                log_silent_failure(
                    logger,
                    "small_stream end log failed (skipped)",
                    exc=e,
                    stage="small_stream",
                    event="end_log_error",
                    model=self.small_model,
                )


# ---------- 工具函数 ----------

def _to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return {"raw": str(obj)}


def _extract_text(data: dict[str, Any]) -> str:
    try:
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return msg.get("content") or ""
    except Exception as e:
        log_silent_failure(
            logger,
            "_extract_text fallback to empty",
            exc=e,
            stage="llm_extract_text",
            event="extract_error",
        )
        return ""


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _should_retry_starved(data: dict[str, Any], eff_max: int | None) -> bool:
    """判断是否因推理思考吃光 max_tokens 而 content 为空，需要加大预算重试。

    DeepSeek 风格模型把推理过程放进独立的 ``reasoning_content``，且计入
    ``max_tokens`` 总预算；若 max_tokens 偏小（如 ReAct 决策的 400），推理可能
    独占全部预算，导致 ``content`` 为空、``finish_reason=length``，工具调用 JSON
    根本没输出。此时把预算翻倍重试一次即可让 JSON 落地。
    """
    if not eff_max or eff_max >= 2048:
        return False
    choices = data.get("choices") or []
    if not choices:
        return False
    if (_extract_text(data) or "").strip():
        return False
    return choices[0].get("finish_reason") == "length"


def parse_tool_call(text: str) -> dict | None:
    """从 LLM 输出中尝试解析 tool call JSON。

    规则：
    1. 整体是合法 JSON 且包含 `tool` 字段 → 命中。
    2. 文本中嵌入 `{...}` JSON 片段 → 命中。
    3. 否则返回 None（视为普通回复）。
    """
    if not text:
        return None
    text = text.strip()
    # 1) 整体 JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and obj.get("tool"):
                return obj
        except Exception as e:
            log_silent_failure(
                logger,
                "parse_tool_call: whole JSON parse failed",
                exc=e,
                stage="llm_parse_tool_call",
                event="whole_json_error",
                text_preview=text[:80],
            )
    # 2) 抽取首个 JSON 片段
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and obj.get("tool"):
                return obj
        except Exception as e:
            log_silent_failure(
                logger,
                "parse_tool_call: embedded JSON parse failed",
                exc=e,
                stage="llm_parse_tool_call",
                event="embedded_json_error",
                snippet_preview=m.group(0)[:80],
            )
    return None


# ---------- 单例 ----------

_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_llm_client() -> None:
    global _client
    _client = None