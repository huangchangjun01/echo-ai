"""回归测试：biz/recall.py 中转录失败占位 ParsedFile 不被当作有效内容。

用户反馈 case：
  上传拉师傅来我家的三年.mp3.mpeg（whisper 模块未安装，转录失败），
  parse_audio 返回了占位 ParsedFile（detail_md="[音频转录失败] ..."，
  transcribeStatus="failed"），但 parse_memory 仍把它当作有效内容喂给
  build_memory_md，导致 LLM 基于占位 + 文件名编造 699 字符的 md。

修复后行为：
  - _is_failed_parse 检测 transcribeStatus="failed"，把这种 ParsedFile
    判定为失败。
  - parse_memory 走 failed_sources 分支：所有源失败 → skip md write；
    部分失败 → 仅在 md 末尾追加"解析失败清单"小节，绝不喂 LLM 拼正文。
"""
from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


def _patch_embed_module():
    fake = types.ModuleType("embedding.models")

    def compute_text_embeddings(texts, device="auto"):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def compute_image_embeddings(images, device="auto"):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    fake.compute_text_embeddings = compute_text_embeddings
    fake.compute_image_embeddings = compute_image_embeddings
    fake.compute_embedding = lambda b: [0.1, 0.2, 0.3, 0.4]
    fake.compute_text_embedding = lambda t: compute_text_embeddings([t])[0]
    fake.load_model = lambda device="auto": (None, None, None)
    fake.warmup = lambda device="auto", batch_size=1: None
    fake.detect_device = lambda preferred="auto": "cpu"
    sys.modules["embedding.models"] = fake


@pytest.fixture(autouse=True)
def _fake_models(monkeypatch):
    _patch_embed_module()
    yield


def _placeholder_audio_parsed(file_name: str = "拉师傅来我家的三年.mp3.mpeg"):
    """构造 parse_audio 在转录失败时返回的占位 ParsedFile。"""
    from parsers.base import ParsedFile, ParsedChunk

    placeholder = f"[音频转录失败] {file_name}"
    return ParsedFile(
        modality="audio",
        text=placeholder,
        chunks=[ParsedChunk(text=placeholder, source=file_name)],
        detail_md=placeholder,  # 与 parse_audio 的实现对齐
        meta={"bytes": 1447645, "transcribeStatus": "failed"},
    )


def test_is_failed_parse_detects_transcribe_failed():
    """核心回归：transcribeStatus="failed" 必须判失败。"""
    from biz.recall import _is_failed_parse

    p = _placeholder_audio_parsed()
    assert _is_failed_parse(p) is True, (
        f"转录失败的占位 ParsedFile 必须判失败，实际 _is_failed_parse 返回 False，"
        f"meta={p.meta}"
    )


def test_is_failed_parse_passes_normal_audio():
    """正常音频 ParsedFile 必须判成功（不被本次修改误伤）。"""
    from biz.recall import _is_failed_parse
    from parsers.base import ParsedFile, ParsedChunk

    p = ParsedFile(
        modality="audio",
        text="你好世界",
        chunks=[ParsedChunk(text="你好世界", source="voice.mp3")],
        detail_md="你好世界，这是真实的转录文本。",
        meta={"bytes": 1000},
    )
    assert _is_failed_parse(p) is False, "正常音频不应判失败"


def test_parse_memory_skips_md_when_all_sources_transcribe_failed(monkeypatch):
    """所有源文件都是转录失败占位 → parse_memory 必须 skip md write（绝不调 LLM）。"""
    from biz import recall as recall_mod

    # 哨兵：build_memory_md 被调用就报错（核心断言）
    def _fail_build(*args, **kwargs):
        raise AssertionError("build_memory_md 不应被调用——所有源都是转录失败占位")

    monkeypatch.setattr(recall_mod, "build_memory_md", _fail_build)
    monkeypatch.setattr(recall_mod, "try_acquire_edit_lock", AsyncMock(return_value=True))
    monkeypatch.setattr(recall_mod, "release_edit_lock", AsyncMock(return_value=True))
    mark_done_mock = AsyncMock()
    monkeypatch.setattr(recall_mod, "mark_done", mark_done_mock)
    monkeypatch.setattr(recall_mod, "upload_bytes", MagicMock())
    monkeypatch.setattr(recall_mod, "get_recall_store", MagicMock())

    sources = [
        {
            "fileKey": "memory/2/3/xxx/yyy.mpeg",
            "fileName": "拉师傅来我家的三年.mp3.mpeg",
            "fileType": 4,  # audio
            "url": "",
        },
    ]

    # 直接 patch _parse_all 跳过实际下载/whisper 调用
    async def _fake_parse_all(sources):
        return [(sources[0]["fileName"], _placeholder_audio_parsed(sources[0]["fileName"]))]

    monkeypatch.setattr(recall_mod, "_parse_all", _fake_parse_all)

    asyncio.run(
        recall_mod.parse_memory(
            user_id="2",
            role_id="3",
            memory_id="44e30b5ff78a4f2ea45af7c0d6e02703",
            topic="拉师傅来家的三年",
            subjective_desc="",
            sources=sources,
        )
    )

    # 断言：parse_status 被标记为 FAILED
    mark_done_calls = mark_done_mock.call_args_list
    failed_called = any(
        call.kwargs.get("parse_status") == recall_mod._PARSE_STATUS_FAILED
        for call in mark_done_calls
    )
    assert failed_called, f"应有 parse_status=FAILED 的 mark_done 调用，实际 {mark_done_calls}"


def test_parse_memory_partial_failure_no_llm_for_placeholder(monkeypatch):
    """混合源：1 个成功 + 1 个转录失败占位 → md 生成只走成功源，绝不喂占位给 LLM。"""
    from biz import recall as recall_mod
    from parsers.base import ParsedFile, ParsedChunk

    # 哨兵：捕获 build_memory_md 的入参，验证 details 里没有占位
    captured: dict = {}

    async def _capture_build_md(**kwargs):
        captured.update(kwargs)
        return "fake md content"

    monkeypatch.setattr(recall_mod, "build_memory_md", _capture_build_md)
    monkeypatch.setattr(recall_mod, "try_acquire_edit_lock", AsyncMock(return_value=True))
    monkeypatch.setattr(recall_mod, "release_edit_lock", AsyncMock(return_value=True))
    monkeypatch.setattr(recall_mod, "mark_done", AsyncMock())
    monkeypatch.setattr(recall_mod, "upload_bytes", AsyncMock())
    monkeypatch.setattr(recall_mod, "get_recall_store", MagicMock())

    ok_audio = ParsedFile(
        modality="audio",
        text="今天和拉师傅一起吃饭",
        chunks=[ParsedChunk(text="今天和拉师傅一起吃饭", source="normal.mp3")],
        detail_md="今天和拉师傅一起吃饭，很开心。",
        meta={"bytes": 1000},
    )
    failed_audio = _placeholder_audio_parsed("拉师傅来我家的三年.mp3.mpeg")

    async def _fake_parse_all(sources):
        return [
            ("normal.mp3", ok_audio),
            ("拉师傅来我家的三年.mp3.mpeg", failed_audio),
        ]

    monkeypatch.setattr(recall_mod, "_parse_all", _fake_parse_all)

    sources = [
        {"fileKey": "k1", "fileName": "normal.mp3", "fileType": 4, "url": ""},
        {"fileKey": "k2", "fileName": "拉师傅来我家的三年.mp3.mpeg", "fileType": 4, "url": ""},
    ]

    asyncio.run(
        recall_mod.parse_memory(
            user_id="2",
            role_id="3",
            memory_id="xxx",
            topic="拉师傅日常",
            subjective_desc="",
            sources=sources,
        )
    )

    # 核心断言：details 里只有正常音频的 detail，没有占位文本
    details = captured.get("details", [])
    assert len(details) >= 1, "至少应有正常音频的 detail"
    for d in details:
        assert "音频转录失败" not in d.get("detail", ""), (
            f"占位 ParsedFile 的 detail 不应进入 build_memory_md，"
            f"实际 details={details}"
        )

    # 失败清单应包含失败的源文件名
    failed_in_md = any(
        "拉师傅来我家的三年.mp3.mpeg" in (d.get("fileName") or "") and "解析失败" in (d.get("fileName") or "")
        for d in details
    )
    # 这里检查 details 中是否有 "解析失败清单" 小节
    has_failed_list = any(
        "解析失败清单" in (d.get("fileName") or "") for d in details
    )
    assert has_failed_list, f"应在 md 中追加'解析失败清单'小节，实际 details={details}"
