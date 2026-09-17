"""Unit tests for procedural mood dispatch during chat_stream delta streaming.

背景(spec §4.3): 当前 mood_update 事件只在 done 后一次 yield（且阈值 0.15 太迟钝）。
本测试验证：
- mood_update 应在 delta 流式期间多次 yield（过程中下发）；
- emotion 标签变化时 yield emotion_change；
- 阈值从 0.15 降到 0.05（更敏感）。

测试策略：mock 掉 LLM / DB / 外部依赖，让 small_stream 输出 N 个 delta chunk。
N ≥ 20 才能触发过程性采样（默认 MOOD_SAMPLE_INTERVAL = 20）。
"""

from __future__ import annotations

from typing import Any


# ---------- Fake LLM clients ----------

class _StreamingFakeLLMClient:
    """Mock LLM 客户端：small_stream 输出 N 个 chunk；ReAct 默认返回 OK（无工具调用）。

    - ``stream_chunks``: small_stream 依次 yield 的 delta 文本片段。
      单 chunk 可以是任意非空字符串（不会触发 <think> 剥离或 URL scheme 改写）。
    - ReAct 决策阶段（small_chat）返回 "OK"，让循环 break → 无 tool_dispatch 干扰。
    """

    def __init__(self, stream_chunks: list[str] | None = None) -> None:
        self._stream_chunks = stream_chunks or []

    async def small_prefix(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        # 规则路径已覆盖 "hello"；保留兜底返回空串。
        return ""

    async def small_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
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
        for chunk in self._stream_chunks:
            # 新契约下 yield 形状是 3-tuple ``(content, reasoning, error)``
            yield chunk, "", None
        # 必须保留 ``return; yield`` 让函数被识别为 async generator（PEP 525），
        # 即便 stream_chunks 非空；万一外部 mock 把它清空也能正常迭代。
        return
        yield  # pragma: no cover


# ---------- 公共上下文 mock ----------

async def _fake_build_chat_context(
    user_id: str,
    query: str,
    *,
    enable_multimodal: bool = True,
    role_id: str = "default",
) -> dict[str, Any]:
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
    return []


async def _fake_build_segments(
    user_id: str,
    role_id: str = "default",
    *,
    l0_memories: list[str] | None = None,
    recent_summaries: list[str] | None = None,
    tool_descriptions: str | None = None,
) -> dict[str, str]:
    return {k: "" for k in (
        "persona", "traits", "mood", "relationship",
        "belief_summary", "l0_memories", "tool_descriptions",
    )}


def _patch_streaming_common(
    monkeypatch,
    *,
    stream_chunks: list[str] | None,
) -> None:
    """注入 streaming mock。

    重要：monkeypatch 必须在 biz.chat 模块命名空间内替换 ``get_llm_client``，
    否则 ``from llm.client import get_llm_client`` 保留在模块里，仍调到真实 LLM。
    """
    import biz.chat as chat_mod
    import biz.recall_search as recall_mod

    monkeypatch.setattr(
        chat_mod,
        "get_llm_client",
        lambda: _StreamingFakeLLMClient(stream_chunks=stream_chunks),
    )
    monkeypatch.setattr(chat_mod, "build_chat_context", _fake_build_chat_context)
    monkeypatch.setattr(chat_mod, "build_segments", _fake_build_segments)
    monkeypatch.setattr(recall_mod, "search_recall_for_chat", _fake_search_recall_for_chat)


# ---------- 工具函数 ----------

def _chunked(chars: str, piece_size: int = 1) -> list[str]:
    """把一段字符串切成 piece_size 字符一组（模拟小模型逐字流式输出）。"""
    return [chars[i : i + piece_size] for i in range(0, len(chars), piece_size)] or [chars]


# ---------- 测试 ----------

async def test_mood_update_emitted_during_stream(monkeypatch):
    """mood_update 应在 delta 流式期间 yield（不只是 done 后一次）。

    验证：
    - small_stream 输出 ≥ 20 个 delta chunk；
    - mood_update 帧在 done **之前** yield（过程中下发），至少 1 次。
    - mood_update schema 字段完整。

    选择 user_msg "我好开心好喜欢好幸福" 让 detect_emotion_fallback 拿到 valence≈0.8，
    intensity=1.0，保证第一次采样的 |delta| > 0.05（新阈值）。
    """
    # 25 个 chunk 字符串足够触发 N=20 的过程性采样
    chunks = _chunked("我好开心好喜欢好幸福" * 5, piece_size=1)
    assert len(chunks) >= 20, f"测试用例需要 ≥20 chunk, got {len(chunks)}"
    _patch_streaming_common(monkeypatch, stream_chunks=chunks)

    from biz.chat import chat_stream

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="1",
        session_id="s1",
        user_msg="我好开心好喜欢好幸福",
        role_id="default",
    ):
        events.append(ev)

    mood_events = [e for e in events if e.get("type") == "mood_update"]
    done_idx = next(i for i, e in enumerate(events) if e.get("type") == "done")

    # 1) 至少 1 个 mood_update 帧
    assert len(mood_events) >= 1, (
        f"expected ≥1 mood_update frame, got 0; events={[e['type'] for e in events]}"
    )

    # 2) 至少 1 个 mood_update 必须在 done 之前 yield（即"过程中"下发）
    before_done = [
        e for e in mood_events if events.index(e) < done_idx
    ]
    assert len(before_done) >= 1, (
        "mood_update 必须在 done 之前 yield 至少一次；"
        f"事件序列={[e['type'] for e in events]}"
    )

    # 3) schema 校验（取第一个过程内 mood_update）
    mu = before_done[0]
    expected_fields = {
        "instantVal",
        "instantIntensity",
        "instantEmotion",
        "shortVal",
        "baselineVal",
        "expression",
        "valence",
    }
    missing = expected_fields - set(mu.keys())
    assert not missing, f"mood_update 缺失字段 {missing}: {mu}"


async def test_emotion_change_emitted_when_label_changes(monkeypatch):
    """emotion 标签变化时应 yield emotion_change（spec §3.2）。

    验证：
    - emotion_change 帧至少 1 次；
    - schema 字段完整（from / to / intensity）。
    - from 是上一时刻的 emotion（默认 neutral），to 是新 emotion。
    """
    # ≥20 chunks 触发 process 采样；强正面情感确保 emotion 从 neutral → happy
    chunks = _chunked("我好开心好喜欢好幸福" * 5, piece_size=1)
    assert len(chunks) >= 20
    _patch_streaming_common(monkeypatch, stream_chunks=chunks)

    from biz.chat import chat_stream

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="1",
        session_id="s1",
        user_msg="我好开心好喜欢好幸福",
        role_id="default",
    ):
        events.append(ev)

    emotion_changes = [e for e in events if e.get("type") == "emotion_change"]
    assert len(emotion_changes) >= 1, (
        f"expected ≥1 emotion_change, got 0; events={[e['type'] for e in events]}"
    )

    # schema 校验
    ec = emotion_changes[0]
    for field in ("from", "to", "intensity"):
        assert field in ec, f"emotion_change missing {field}: {ec}"
    assert isinstance(ec["intensity"], (int, float)), (
        f"emotion_change.intensity 必须是数值, got {type(ec['intensity'])}: {ec}"
    )

    # 默认起点 emotion = "neutral"；强正面情感至少应触达 "happy"
    assert ec["from"] == "neutral", f"默认起点应为 neutral, got {ec['from']}"
    assert ec["to"] in {"happy", "excited", "neutral"}, (
        f"emotion_change.to 须是合法 emotion 标签, got {ec['to']}"
    )
    assert ec["from"] != ec["to"], (
        f"emotion 标签必须真的发生变化: from={ec['from']} to={ec['to']}"
    )


async def test_mood_threshold_lowered_to_005(monkeypatch):
    """mood_update 阈值从 0.15 降到 0.05（spec §4.3）。

    选择 "你好呀"（detect_emotion_fallback 拿到 valence≈0.2, intensity=0.3）；
    新阈值 0.05 下首次采样的 |delta|≈0.06 应触发 mood_update。
    若阈值仍为 0.15，则不会触发（0.06 < 0.15）——本测试会失败。

    注意:本测试依赖 ``character.mood.detect_emotion_fallback`` 的关键词权重。
    "你好呀" 不在正/负向词表里 → valence=0, 但"好"字在正向词表里命中 1 次(+0.2)。
    改 detect_emotion_fallback 的 keyword 列表后,此处 valence 与阈值裕度都会变,
    评估「0.06 触发」是否仍成立需要重算——别忘记同步调整本测试。
    """
    # ≥20 chunks 触发 process 采样
    chunks = _chunked("字" * 25, piece_size=1)
    assert len(chunks) >= 20
    _patch_streaming_common(monkeypatch, stream_chunks=chunks)

    from biz.chat import chat_stream

    events: list[dict[str, Any]] = []
    async for ev in chat_stream(
        user_id="1",
        session_id="s1",
        user_msg="你好呀",  # valence≈0.2 → |delta|≈0.06，新阈值下应触发
        role_id="default",
    ):
        events.append(ev)

    mood_events = [e for e in events if e.get("type") == "mood_update"]
    done_idx = next(i for i, e in enumerate(events) if e.get("type") == "done")
    before_done = [e for e in mood_events if events.index(e) < done_idx]

    assert len(before_done) >= 1, (
        "阈值 0.05 下, '你好呀' 的过程性 mood_update 应该 yield；"
        f"实际 events={[e['type'] for e in events]}"
    )