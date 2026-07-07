"""Unit tests for llm.intent.classify_intent.

Mock LLM client 不依赖真实小模型，验证：
- 正常 JSON 解析到 5 类
- 6 种 fallback 路径各自落到 Intent.CHAT 并打对应 source 标签
"""

from __future__ import annotations

import asyncio

import pytest


class _FakeLLMClient:
    """按预定 schedule 返回 small_prefix 结果（支持 sleep 模拟超时）。"""

    def __init__(self, text: str = "", *, delay: float = 0.0):
        self.text = text
        self.delay = delay
        self.calls = 0

    async def small_prefix(self, messages, *, max_tokens=None, temperature=None):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.text


def _patch_client(monkeypatch, client: _FakeLLMClient) -> None:
    from llm import intent

    monkeypatch.setattr(intent, "get_llm_client", lambda: client)


async def test_classify_chat(monkeypatch):
    from llm.intent import Intent, classify_intent

    _patch_client(monkeypatch, _FakeLLMClient(text='{"intent":"chat","reason":"闲聊"}'))
    res = await classify_intent("今天好累")
    assert res.intent is Intent.CHAT
    assert res.source == "llm"
    assert res.duration_ms >= 0


async def test_classify_image_search(monkeypatch):
    from llm.intent import Intent, classify_intent

    _patch_client(monkeypatch, _FakeLLMClient(text='{"intent":"image_search","reason":"找图"}'))
    res = await classify_intent("找到黄色小狗的照片")
    assert res.intent is Intent.IMAGE_SEARCH
    assert res.source == "llm"


async def test_classify_recall(monkeypatch):
    from llm.intent import Intent, classify_intent

    _patch_client(monkeypatch, _FakeLLMClient(text='{"intent":"recall","reason":"回忆"}'))
    res = await classify_intent("我叫什么名字")
    assert res.intent is Intent.RECALL


async def test_classify_text_search(monkeypatch):
    from llm.intent import Intent, classify_intent

    _patch_client(monkeypatch, _FakeLLMClient(text='{"intent":"text_search","reason":"文本"}'))
    res = await classify_intent("找那段笔记")
    assert res.intent is Intent.TEXT_SEARCH


async def test_classify_doc_search(monkeypatch):
    from llm.intent import Intent, classify_intent

    _patch_client(monkeypatch, _FakeLLMClient(text='{"intent":"doc_search","reason":"文档"}'))
    res = await classify_intent("在 RAG 库找简历")
    assert res.intent is Intent.DOC_SEARCH


async def test_classify_timeout_returns_chat(monkeypatch):
    from config.config import get_settings
    from llm.intent import Intent, classify_intent

    # 故意延迟 5s，但 timeout=1500ms
    _patch_client(monkeypatch, _FakeLLMClient(text="", delay=5.0))
    settings = get_settings().memory
    original_timeout = settings.intent_timeout_ms
    settings.intent_timeout_ms = 200  # 调小以加速测试
    try:
        res = await classify_intent("任意消息")
        assert res.intent is Intent.CHAT
        assert res.source == "fallback_timeout"
    finally:
        settings.intent_timeout_ms = original_timeout


async def test_classify_invalid_label_returns_chat(monkeypatch):
    from llm.intent import Intent, classify_intent

    _patch_client(monkeypatch, _FakeLLMClient(text='{"intent":"unknown_label"}'))
    res = await classify_intent("任意消息")
    assert res.intent is Intent.CHAT
    assert res.source == "fallback_invalid_label"


async def test_classify_parse_error_returns_chat(monkeypatch):
    from llm.intent import Intent, classify_intent

    _patch_client(monkeypatch, _FakeLLMClient(text="我不是 JSON"))
    res = await classify_intent("任意消息")
    assert res.intent is Intent.CHAT
    assert res.source == "fallback_parse_error"


async def test_classify_empty_msg_returns_chat(monkeypatch):
    from llm.intent import Intent, classify_intent

    _patch_client(monkeypatch, _FakeLLMClient(text='{"intent":"chat"}'))
    res = await classify_intent("   ")
    assert res.intent is Intent.CHAT
    assert res.source == "fallback_empty"
    # 不应触发 LLM 调用
    assert res.duration_ms == 0.0


async def test_classify_disabled_returns_chat(monkeypatch):
    from config.config import get_settings
    from llm.intent import Intent, classify_intent

    settings = get_settings().memory
    original = settings.intent_classifier_enabled
    settings.intent_classifier_enabled = False
    client = _FakeLLMClient(text='{"intent":"image_search"}')
    _patch_client(monkeypatch, client)
    try:
        res = await classify_intent("任意消息")
        assert res.intent is Intent.CHAT
        assert res.source == "fallback_disabled"
        assert client.calls == 0  # 完全跳过 LLM
    finally:
        settings.intent_classifier_enabled = original


async def test_classify_exception_returns_chat(monkeypatch):
    """small_prefix 抛异常时也要回退到 chat，不阻断对话。"""

    class _BrokenClient:
        async def small_prefix(self, *a, **kw):
            raise RuntimeError("upstream boom")

    from llm import intent
    from llm.intent import Intent, classify_intent

    monkeypatch.setattr(intent, "get_llm_client", lambda: _BrokenClient())
    res = await classify_intent("任意消息")
    assert res.intent is Intent.CHAT
    assert res.source == "fallback_error"


async def test_intent_values():
    """枚举值是稳定的字符串契约，前端/SSE 依赖。"""
    from llm.intent import Intent

    assert Intent.CHAT.value == "chat"
    assert Intent.RECALL.value == "recall"
    assert Intent.TEXT_SEARCH.value == "text_search"
    assert Intent.IMAGE_SEARCH.value == "image_search"
    assert Intent.DOC_SEARCH.value == "doc_search"
