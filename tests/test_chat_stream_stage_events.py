"""Unit tests for stage event contract in biz.chat.chat_stream.

这些测试验证 SSE v2 阶段事件（intent / recall_search / tool_dispatch /
model_reasoning / answer）的 yield 边界。

注意：纯意图（CHAT）消息"hello"应该走规则快速路径，无工具调用，
所以 tool_dispatch 阶段不应出现。
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


class _FakeLLMClient:
    """极简 LLM 客户端 mock，支撑 chat_stream 完成流式而不真发请求。

    - classify_intent("hello") 走规则路径，**不**调用 LLM。
    - _resolve_tools 内 small_chat 默认返回 "OK"，**不**触发任何工具调用；
      构造时传入 ``tool_call_payload`` 则第一次决策返回工具调用 JSON，触发 tool_dispatch 阶段。
    - small_stream 不产生任何 chunk，让 reply 干净结束。
    """

    def __init__(self, tool_call_payload: dict | None = None) -> None:
        self._tool_call_payload = tool_call_payload

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
        # 配置过 tool_call_payload → 第一次决策返回工具调用 JSON；
        # 之后返回 "OK" 让循环退出（模拟 _resolve_tools 内部 break）。
        if self._tool_call_payload is not None:
            payload, self._tool_call_payload = self._tool_call_payload, None
            return {
                "choices": [
                    {"message": {"content": json.dumps(payload), "finish_reason": "tool_calls"}}
                ],
                "usage": {},
            }
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
        # 空 async generator：reply 阶段直接走 final_text 清理路径，不产生任何 chunk。
        # 用 ``return`` + 不可达 ``yield`` 把函数标记为 async generator（PEP 525），
        # 让上游 ``async for small_stream(...)`` 可以正常迭代；
        # 若去掉 ``yield`` 会变成普通 async 函数返回 None，触发 TypeError: 'async for' received non-async-iterator。
        # ``yield`` 行**永远不会被执行**（函数在 ``return`` 处就结束了），
        # 仅作类型标记存在；``# pragma: no cover`` 让覆盖率工具忽略这条永远不可达的代码。
        # 新契约下 yield 形状是 3-tuple ``(content, reasoning, error)``；
        # 本测试只需「不产出任何 chunk」，元组形状无关紧要（async for 不会触发解包）。
        return
        yield  # pragma: no cover  # 永远不可达；仅作 async generator 类型标记（PEP 525）


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


def _patch_common(
    monkeypatch,
    *,
    tool_call_payload: dict | None = None,
    recall_search_side_effect: Exception | None = None,
    dispatch_tool_side_effect: Exception | None = None,
) -> None:
    """注入 mock，屏蔽真实 LLM / DB / Embedding 调用。

    参数全部可选，测试按需打开：
    - tool_call_payload: 让 _FakeLLMClient 在 small_chat 第一次返回工具调用 JSON，
      触发 tool_dispatch 阶段路径。
    - recall_search_side_effect: 让 ``biz.recall_search.search_recall_for_chat``
      抛指定异常，验证 recall_search end 仍能正常 yield。
    - dispatch_tool_side_effect: 让 ``biz.chat.dispatch_tool`` 抛指定异常，
      验证 tool_dispatch end 仍能正常 yield（且 break 出循环）。

    重要：必须 monkeypatch **biz.chat.get_llm_client**（而非 llm.client.get_llm_client），
    否则 ``from llm.client import get_llm_client`` 在 biz.chat 模块命名空间里保留旧引用，
    仍会调到真实 LLM。
    """
    import biz.chat as chat_mod

    monkeypatch.setattr(
        chat_mod, "get_llm_client", lambda: _FakeLLMClient(tool_call_payload)
    )
    monkeypatch.setattr(chat_mod, "build_chat_context", _fake_build_chat_context)
    monkeypatch.setattr(chat_mod, "build_segments", _fake_build_segments)

    async def _fake_persona_segment(user_id: str, role_id: str = "default") -> str:
        """空人格段：让 context 帧回落 ctx.persona，既有断言保持稳定。"""
        return ""

    monkeypatch.setattr(chat_mod, "load_persona_segment", _fake_persona_segment)

    async def _recall_or_raise(*args, **kwargs):
        if recall_search_side_effect is not None:
            raise recall_search_side_effect
        return []

    import biz.recall_search as recall_mod

    monkeypatch.setattr(recall_mod, "search_recall_for_chat", _recall_or_raise)

    if dispatch_tool_side_effect is not None:
        async def _dispatch_or_raise(*args, **kwargs):
            raise dispatch_tool_side_effect

        monkeypatch.setattr(chat_mod, "dispatch_tool", _dispatch_or_raise)


def _stage_names(events: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """抽出 stage 事件的 (name, state) 序列，便于断言。"""
    return [(e["name"], e["state"]) for e in events if e.get("type") == "stage"]


async def test_stage_events_chat_path_no_tools(monkeypatch):
    """纯意图对话应产出 intent / recall_search / model_reasoning / answer 四阶段，无 tool_dispatch。

    注意：本测试命名「chat_path_no_tools」——完整覆盖 CHAT 路径下的所有阶段，
    包括无条件 yield 的 recall_search。
    """
    _patch_common(monkeypatch)

    from biz.chat import chat_stream, StageName, StageState

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="1",
        session_id="s1",
        user_msg="hello",
        role_id="default",
    ):
        events.append(ev)

    names = _stage_names(events)

    # 必须出现的 8 个 stage 帧（每个阶段 start + end）
    expected_pairs = [
        (StageName.INTENT, StageState.START),
        (StageName.INTENT, StageState.END),
        (StageName.RECALL_SEARCH, StageState.START),
        (StageName.RECALL_SEARCH, StageState.END),
        (StageName.MODEL_REASONING, StageState.START),
        (StageName.MODEL_REASONING, StageState.END),
        (StageName.ANSWER, StageState.START),
        (StageName.ANSWER, StageState.END),
    ]
    for pair in expected_pairs:
        assert pair in names, f"missing stage event {pair}: got {names}"

    # tool_dispatch 必须被跳过（CHAT 路径，无工具调用）
    assert (StageName.TOOL_DISPATCH, StageState.START) not in names
    assert (StageName.TOOL_DISPATCH, StageState.END) not in names

    # 阶段顺序约定：intent → recall_search → ... → model_reasoning → answer
    ordered_names = [n for n, _ in names]
    assert ordered_names.index(StageName.INTENT) < ordered_names.index(StageName.RECALL_SEARCH)
    assert ordered_names.index(StageName.RECALL_SEARCH) < ordered_names.index(StageName.MODEL_REASONING)
    assert ordered_names.index(StageName.MODEL_REASONING) < ordered_names.index(StageName.ANSWER)
    done_idx = next(
        i for i, e in enumerate(events) if e.get("type") == "done"
    )
    answer_end_idx = next(
        i for i, e in enumerate(events)
        if e.get("type") == "stage"
        and e.get("name") == StageName.ANSWER
        and e.get("state") == StageState.END
    )
    assert answer_end_idx < done_idx


async def test_stage_events_have_ts_ms(monkeypatch):
    """每个 stage 事件必须有 ts_ms 字段（int，便于前端计算耗时）。"""
    _patch_common(monkeypatch)

    from biz.chat import chat_stream

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="1",
        session_id="s1",
        user_msg="hello",
        role_id="default",
    ):
        events.append(ev)

    stage_events = [e for e in events if e.get("type") == "stage"]
    assert len(stage_events) >= 8, f"expected >=8 stage events, got {len(stage_events)}"
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
        user_id="1",
        session_id="s1",
        user_msg="hello",
        role_id="default",
    ):
        events.append(ev)

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


async def test_tool_dispatch_stage_events_when_llm_calls_tool(monkeypatch):
    """当 LLM 决策返回工具调用 JSON 时，必须 yield 一对 tool_dispatch stage 事件。

    验证：
    - tool_dispatch.start 与 end 配对出现；
    - start 事件带 ``iter=0`` 与 ``tool=<tool_name>``；
    - 阶段顺序：tool_dispatch.start 在 dispatch 之前 yield；
    - 之后 model_reasoning / answer 仍正常 yield。
    """
    payload = {"tool": "search_memory", "args": {"query": "hello"}}
    _patch_common(monkeypatch, tool_call_payload=payload)

    from biz.chat import chat_stream, StageName, StageState

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="1",
        session_id="s1",
        user_msg="hello",
        role_id="default",
    ):
        events.append(ev)

    # 必须出现 tool_dispatch start/end 各 1 次
    tool_starts = [
        e for e in events
        if e.get("type") == "stage"
        and e.get("name") == StageName.TOOL_DISPATCH
        and e.get("state") == StageState.START
    ]
    tool_ends = [
        e for e in events
        if e.get("type") == "stage"
        and e.get("name") == StageName.TOOL_DISPATCH
        and e.get("state") == StageState.END
    ]
    assert len(tool_starts) == 1, f"expected 1 tool_dispatch start, got {len(tool_starts)}: {_stage_names(events)}"
    assert len(tool_ends) == 1, f"expected 1 tool_dispatch end, got {len(tool_ends)}: {_stage_names(events)}"

    # start 事件必须带 iter 与 tool 扩展字段
    start_ev = tool_starts[0]
    assert start_ev.get("iter") == 0, f"tool_dispatch start iter must be 0: {start_ev}"
    assert start_ev.get("tool") == "search_memory", (
        f"tool_dispatch start tool must be 'search_memory': {start_ev}"
    )

    # 顺序约束：tool_dispatch.start < tool_dispatch.end < model_reasoning.start
    start_idx = events.index(tool_starts[0])
    end_idx = events.index(tool_ends[0])
    assert start_idx < end_idx

    # tool_dispatch.end 仍要在 model_reasoning.start 之前
    mr_start_idx = next(
        i for i, e in enumerate(events)
        if e.get("type") == "stage"
        and e.get("name") == StageName.MODEL_REASONING
        and e.get("state") == StageState.START
    )
    assert end_idx < mr_start_idx, (
        "tool_dispatch.end 必须在 model_reasoning.start 之前 yield"
    )


async def test_recall_search_end_yields_even_when_search_raises(monkeypatch):
    """``search_recall_for_chat`` 抛异常时，recall_search.end 仍必须 yield。

    验证「try/finally」修复：客户端断连或检索异常都不能让前端卡在「recall_search 进行中」。
    """
    _patch_common(
        monkeypatch,
        recall_search_side_effect=RuntimeError("recall backend down"),
    )

    from biz.chat import chat_stream, StageName, StageState

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="1",
        session_id="s1",
        user_msg="hello",
        role_id="default",
    ):
        events.append(ev)

    # recall_search start 与 end 都必须出现
    recall_starts = [
        e for e in events
        if e.get("type") == "stage"
        and e.get("name") == StageName.RECALL_SEARCH
        and e.get("state") == StageState.START
    ]
    recall_ends = [
        e for e in events
        if e.get("type") == "stage"
        and e.get("name") == StageName.RECALL_SEARCH
        and e.get("state") == StageState.END
    ]
    assert len(recall_starts) == 1, f"recall_search.start 缺失：{recall_starts}"
    assert len(recall_ends) == 1, f"recall_search.end 在异常后缺失：{recall_ends}"
    assert events.index(recall_starts[0]) < events.index(recall_ends[0])


async def test_tool_dispatch_end_yields_even_when_tool_raises(monkeypatch):
    """``dispatch_tool`` 抛异常时，tool_dispatch.end 仍必须 yield。

    验证：异常路径下 start/end 配对仍平衡；循环 break 后 model_reasoning 阶段继续。
    """
    payload = {"tool": "search_memory", "args": {"query": "hello"}}
    _patch_common(
        monkeypatch,
        tool_call_payload=payload,
        dispatch_tool_side_effect=RuntimeError("tool dispatch failed"),
    )

    from biz.chat import chat_stream, StageName, StageState

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="1",
        session_id="s1",
        user_msg="hello",
        role_id="default",
    ):
        events.append(ev)

    tool_starts = [
        e for e in events
        if e.get("type") == "stage"
        and e.get("name") == StageName.TOOL_DISPATCH
        and e.get("state") == StageState.START
    ]
    tool_ends = [
        e for e in events
        if e.get("type") == "stage"
        and e.get("name") == StageName.TOOL_DISPATCH
        and e.get("state") == StageState.END
    ]
    assert len(tool_starts) == 1, "tool_dispatch.start 必须出现"
    assert len(tool_ends) == 1, "tool_dispatch.end 在 dispatch 抛错后仍必须 yield"
    assert events.index(tool_starts[0]) < events.index(tool_ends[0])

    # 异常路径下循环 break，后续阶段仍正常 yield
    names = _stage_names(events)
    assert (StageName.MODEL_REASONING, StageState.START) in names
    assert (StageName.ANSWER, StageState.END) in names


# ---------- §3.5 错误事件保证 ----------

class _BoomCompletions:
    """让 ``completions.create(stream=True)`` 抛错，模拟 LLM 流式异常。

    非流式调用（ReAct 决策）返回 ``"OK"``，让 ReAct 循环直接退出，不引入工具调用干扰。
    """

    async def create(self, *args, **kwargs):
        if kwargs.get("stream"):
            raise RuntimeError("LLM stream backend down")
        return {
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {},
        }


class _BoomChat:
    completions = _BoomCompletions()


class _BoomSmallClient:
    chat = _BoomChat()


def _make_erroring_llm_client():
    """构造一个真实 LLMClient，但 monkey-patch 其 ``_small_client`` 让 stream 抛错。

    使用真实 LLMClient（而不是 FakeLLMClient）是为了让真实 ``small_stream`` 的
    内部 ``try/except`` 接管异常——这样测试覆盖的是「真实 small_stream 把异常
    yield 成错误信号」这一关键路径，而不是「fake 已经把异常转成 error tuple」。
    """
    from llm.client import LLMClient

    # 跳过 __init__（避免连真实 LLM 创建 AsyncOpenAI 客户端）
    client = LLMClient.__new__(LLMClient)
    client._small_client = _BoomSmallClient()  # type: ignore[attr-defined]
    client.small_model = "mock-small-model"
    client.small_max_tokens = 1024
    client.small_temperature = 0.7
    return client


async def test_llm_error_yields_error_frame(monkeypatch):
    """小模型流式异常应被 yield 出去，不静默吞掉（spec §3.5 错误事件保证）。

    验证：
    - 至少一个 ``error`` 帧，``code == "llm_stream_failed"``；
    - ``error`` 帧之后**仍** yield ``done`` 帧（``full == ""``），让前端能正确进入完成态；
    - ``error`` 帧在 ``done`` 帧之前（保证前端先看到错误状态再退出流式态）。
    """
    import biz.chat as chat_mod
    import biz.recall_search as recall_mod

    monkeypatch.setattr(chat_mod, "get_llm_client", _make_erroring_llm_client)
    monkeypatch.setattr(chat_mod, "build_chat_context", _fake_build_chat_context)
    monkeypatch.setattr(chat_mod, "build_segments", _fake_build_segments)

    async def _fake_persona_segment(user_id: str, role_id: str = "default") -> str:
        """空人格段：让 context 帧回落 ctx.persona。"""
        return ""

    monkeypatch.setattr(chat_mod, "load_persona_segment", _fake_persona_segment)
    monkeypatch.setattr(recall_mod, "search_recall_for_chat", _fake_search_recall_for_chat)

    from biz.chat import chat_stream

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="1",
        session_id="s1",
        user_msg="hello",
        role_id="default",
    ):
        events.append(ev)

    error_frames = [e for e in events if e.get("type") == "error"]
    done_frames = [e for e in events if e.get("type") == "done"]

    # 关键断言：error 帧 ≥1 + done 帧 == 1（§3.5 错误事件保证）
    assert len(error_frames) >= 1, (
        f"expected ≥1 error frame, got 0; events={events}"
    )
    assert len(done_frames) == 1, (
        f"expected exactly 1 done frame, got {len(done_frames)}; events={events}"
    )

    # error 帧字段契约
    err = error_frames[0]
    assert err.get("code") == "llm_stream_failed", (
        f"error 帧 code 必须为 'llm_stream_failed': {err}"
    )
    assert err.get("error"), f"error 帧必须有 error 字段: {err}"

    # done 帧契约：error 之后仍 yield done，且 full 为空串
    done = done_frames[0]
    assert done.get("full") == "", (
        f"error 后 done.full 必须为空串: {done}"
    )

    # 顺序约束：error 必须在 done 之前（前端先看到错误状态再退出流式态）
    assert events.index(error_frames[0]) < events.index(done_frames[0]), (
        f"error 帧必须在 done 帧之前 yield：error@{events.index(error_frames[0])} done@{events.index(done_frames[0])}"
    )


async def test_context_frame_uses_effective_role_persona(monkeypatch):
    """胶囊人格展示 = 实际注入人格：角色级 persona_segment 优先于 ctx.persona（DEFAULT）。"""
    _patch_common(monkeypatch)

    import biz.chat as chat_mod

    async def _role_persona(user_id: str, role_id: str = "default") -> str:
        return "【身份】小暖：22岁女生，喜欢小狗"

    monkeypatch.setattr(chat_mod, "load_persona_segment", _role_persona)

    from biz.chat import chat_stream

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="1",
        session_id="s1",
        user_msg="hello",
        role_id="3",
    ):
        events.append(ev)

    ctx_frames = [e for e in events if e.get("type") == "context"]
    assert len(ctx_frames) == 1, f"expected exactly 1 context frame, got {ctx_frames}"
    frame = ctx_frames[0]
    assert frame["persona"] == "【身份】小暖：22岁女生，喜欢小狗"
    assert frame["persona_len"] == len("【身份】小暖：22岁女生，喜欢小狗")


async def test_context_frame_carries_extended_fields(monkeypatch):
    """context 帧方案 A 扩展：persona 全文 / l0_items / l1_items 必须随帧下发。

    前端意图阶段胶囊 popover 依赖这三个字段展示「真实注入内容」；
    只发计数（persona_len / l0_count / l1_count）会让胶囊显示为空。
    """
    _patch_common(monkeypatch)

    async def _rich_ctx(
        user_id: str,
        query: str,
        *,
        enable_multimodal: bool = True,
        role_id: str = "default",
    ) -> dict[str, Any]:
        return {
            "persona": "测试人格：你叫 Echo，温暖、耐心、有同理心。",
            "l0_memories": ["L0-1 用户喜欢 Python", "L0-2 用户从事后端开发"],
            "recent_summaries": [
                "[L2] 父摘要：用户是 Python 后端工程师",
                "[L1] 昨天聊了 Go 并发",
                "[L1] 今天聊了 Vue 前端",
            ],
            "l1_hits": [],
        }

    import biz.chat as chat_mod

    monkeypatch.setattr(chat_mod, "build_chat_context", _rich_ctx)

    from biz.chat import chat_stream

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="1",
        session_id="s1",
        user_msg="hello",
        role_id="default",
    ):
        events.append(ev)

    ctx_frames = [e for e in events if e.get("type") == "context"]
    assert len(ctx_frames) == 1, f"expected exactly 1 context frame, got {ctx_frames}"
    frame = ctx_frames[0]

    # 计数与内容本体必须同时下发
    assert frame["persona"] == "测试人格：你叫 Echo，温暖、耐心、有同理心。"
    assert frame["persona_len"] == len("测试人格：你叫 Echo，温暖、耐心、有同理心。")
    assert frame["l0_items"] == ["L0-1 用户喜欢 Python", "L0-2 用户从事后端开发"]
    assert frame["l0_count"] == 2
    assert frame["l1_items"] == ["[L1] 昨天聊了 Go 并发", "[L1] 今天聊了 Vue 前端"]
    assert frame["l1_count"] == 2
    assert frame["l2_items"] == ["[L2] 父摘要：用户是 Python 后端工程师"]
    assert frame["l2_count"] == 1

