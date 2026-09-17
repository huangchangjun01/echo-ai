"""Unit tests for stage event contract in biz.chat.chat_stream.

这些测试验证 SSE v2 阶段事件（intent / recall_search / tool_dispatch /
model_reasoning / answer）的 yield 边界。

注意：纯意图（CHAT）消息"hello"应该走规则快速路径，无工具调用，
所以 tool_dispatch 阶段不应出现。
"""

from __future__ import annotations

from typing import Any


class _FakeLLMClient:
    """极简 LLM 客户端 mock，支撑 chat_stream 完成流式而不真发请求。

    - classify_intent("hello") 走规则路径，**不**调用 LLM。
    - _resolve_tools 内 small_chat 返回 "OK"，**不**触发任何工具调用。
    - small_stream 不产生任何 chunk，让 reply 干净结束。
    """

    def __init__(self) -> None:
        self.small_chat_calls: int = 0

    async def small_prefix(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        # 规则路径已覆盖 "hello"，这里不会被调用；保留为兜底。
        return ""

    async def small_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        # ReAct 决策：返回 "OK" 表示不要调工具，break 出 _resolve_tools 循环。
        self.small_chat_calls += 1
        return {
            "choices": [
                {"message": {"content": "OK"}, "finish_reason": "stop"}
            ],
            "usage": {},
        }

    async def small_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        # 不产生任何 chunk：reply 阶段直接走 final_text 清理路径，
        # chat_stream 会优雅地 yield {"type": "done", "full": ""}。
        if False:  # 让它成为 async generator
            yield ("", "")


async def _fake_build_chat_context(
    user_id: str,
    query: str,
    *,
    enable_multimodal: bool = True,
    role_id: str = "default",
) -> dict[str, Any]:
    """空上下文：没有人格 / L0 / L1 / 资源。"""
    return {
        "persona": "",
        "l0_memories": [],
        "recent_summaries": [],
        "l1_hits": [],
    }


async def _fake_search_recall_for_chat(
    user_id: str,
    role_id: str,
    message: str,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """空回忆：让 recall_search 阶段正常完成但无 hits。"""
    return []


async def _fake_build_segments(
    user_id: str,
    role_id: str = "default",
    *,
    l0_memories: list[str] | None = None,
    recent_summaries: list[str] | None = None,
    tool_descriptions: str | None = None,
) -> dict[str, str]:
    """空 7 段：避免真实 DB 查询（persona / traits / mood / relationship）。"""
    return {k: "" for k in (
        "persona", "traits", "mood", "relationship",
        "belief_summary", "l0_memories", "tool_descriptions",
    )}


def _patch_common(monkeypatch) -> None:
    """注入 mock，屏蔽真实 LLM / DB / Embedding 调用。"""
    # LLM client
    from llm import client as client_mod

    monkeypatch.setattr(client_mod, "get_llm_client", lambda: _FakeLLMClient())

    # chat_stream 内模块级导入的符号
    import biz.chat as chat_mod

    monkeypatch.setattr(chat_mod, "build_chat_context", _fake_build_chat_context)
    monkeypatch.setattr(chat_mod, "build_segments", _fake_build_segments)

    # chat_stream 内函数级导入的符号（延迟 import 每次拿新引用）
    import biz.recall_search as recall_mod

    monkeypatch.setattr(recall_mod, "search_recall_for_chat", _fake_search_recall_for_chat)


def _stage_names(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """抽出 stage 事件的 (name, state) 序列，便于断言。"""
    return [(e["name"], e["state"]) for e in events if e.get("type") == "stage"]


async def test_stage_events_intent_only(monkeypatch):
    """纯意图对话应该产出 intent / model_reasoning / answer 三个阶段，无 tool_dispatch。"""
    _patch_common(monkeypatch)

    from biz.chat import chat_stream

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="u1",
        session_id="s1",
        user_msg="hello",
        role_id="default",
    ):
        events.append(ev)

    names = _stage_names(events)

    # 必须出现的 6 个 stage 帧
    assert ("intent", "start") in names
    assert ("intent", "end") in names
    assert ("model_reasoning", "start") in names
    assert ("model_reasoning", "end") in names
    assert ("answer", "start") in names
    assert ("answer", "end") in names

    # tool_dispatch 必须被跳过（CHAT 路径，无工具调用）
    assert ("tool_dispatch", "start") not in names
    assert ("tool_dispatch", "end") not in names

    # 阶段顺序约定：intent → recall_search → ... → model_reasoning → answer
    # answer 必须出现在 model_reasoning.end 之后、done 之前
    ordered_names = [n for n, _ in names]
    assert ordered_names.index("model_reasoning") < ordered_names.index("answer")
    done_idx = next(
        i for i, e in enumerate(events) if e.get("type") == "done"
    )
    answer_end_idx = next(
        i for i, e in enumerate(events)
        if e.get("type") == "stage" and e.get("name") == "answer" and e.get("state") == "end"
    )
    assert answer_end_idx < done_idx


async def test_stage_events_have_ts_ms(monkeypatch):
    """每个 stage 事件必须有 ts_ms 字段（int，便于前端计算耗时）。"""
    _patch_common(monkeypatch)

    from biz.chat import chat_stream

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="u1",
        session_id="s1",
        user_msg="hello",
        role_id="default",
    ):
        events.append(ev)

    stage_events = [e for e in events if e.get("type") == "stage"]
    assert len(stage_events) >= 6, f"expected >=6 stage events, got {len(stage_events)}"
    for ev in stage_events:
        assert "ts_ms" in ev, f"stage event missing ts_ms: {ev}"
        assert isinstance(ev["ts_ms"], int), f"ts_ms must be int: {ev}"
        assert ev["ts_ms"] > 0, f"ts_ms must be positive: {ev}"


async def test_stage_events_pairs_balanced(monkeypatch):
    """同一 name 的 start 必须在 end 之前出现，且成对。"""
    _patch_common(monkeypatch)

    from biz.chat import chat_stream

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="u1",
        session_id="s1",
        user_msg="hello",
        role_id="default",
    ):
        events.append(ev)

    from collections import defaultdict

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"start": 0, "end": 0}
    )
    for ev in events:
        if ev.get("type") == "stage":
            counts[ev["name"]][ev["state"]] += 1

    # 出现的 stage 必须 start 和 end 配对（数量相等）
    for name, c in counts.items():
        assert c["start"] == c["end"], (
            f"stage {name} start/end 不配对: start={c['start']} end={c['end']}"
        )