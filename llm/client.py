"""LLM 客户端：OpenAI 兼容协议。

- 大模型：用于深度续写 + 记忆整合（普通 chat / 流式）。
- 小模型（情感微模型）：用于快速前缀生成（首字 < 200ms），仅取前若干 token。
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
    ) -> AsyncIterator[str]:
        """流式 chat completion，逐 chunk 返回增量文本。"""
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
                    delta = chunk.choices[0].delta.content
                    if delta:
                        if first_delta_at is None:
                            first_delta_at = time.perf_counter()
                        delta_count += 1
                        out_chars += len(delta)
                        yield delta
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

    # ---------- 小模型（情感微模型 / 快速前缀） ----------

    async def small_prefix(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """小模型快速生成前缀文本（首字延迟敏感）。

        设计要点：
        - 使用小 `max_tokens` 限制（默认 64），控制首屏响应时间。
        - 失败时返回空串，让大模型独立完成续写，不阻断流程。
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
            )
            data = _to_dict(resp)
            text = _extract_text(data) or ""
            logger.info(
                "small_prefix ok",
                extra=merge_extra(
                    stage="small_prefix",
                    event="ok",
                    model=self.small_model,
                    msg_count=len(messages),
                    out_len=len(text),
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                ),
            )
            return text
        except Exception as e:
            log_exception(
                logger,
                "small_prefix failed",
                exc=e,
                level=logging.WARNING,
                stage="small_prefix",
                event="error",
                model=self.small_model,
                msg_count=len(messages),
                temperature=eff_temp,
                max_tokens=eff_max,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            )
            return ""

    async def small_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """小模型流式输出（用于级联）。"""
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
                    delta = chunk.choices[0].delta.content
                    if delta:
                        if first_delta_at is None:
                            first_delta_at = time.perf_counter()
                        delta_count += 1
                        out_chars += len(delta)
                        yield delta
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