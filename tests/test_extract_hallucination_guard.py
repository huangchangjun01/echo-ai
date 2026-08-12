"""防幻觉测试：当转录失败占位文本 + 无 desc 时，extract_from_file 不应触发 LLM 抽取。

对应用户反馈：
  parse_memory 的解析结果是 LLM 基于「拉师傅来我家的三年.mp3.mpeg」文件名编造的
  推测性内容（"宠物相处三年的回忆"等），而非真实音频转录内容。

修复方案：
1. extract_from_file 检测占位文本 + 空 desc → 直接跳过，不调 LLM
2. FILE_MEMORY_EXTRACT_SYSTEM 加入禁止幻觉的硬性原则作为兜底
"""
from __future__ import annotations

import asyncio
import sys
import types

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


def test_extract_skips_when_placeholder_and_no_desc(monkeypatch):
    """回归测试：转录失败占位 + 无 desc → extract_from_file 直接返回空，不调 LLM。"""
    from memory import extractor as extractor_mod

    # 哨兵：若 _llm_extract_file 被调用，本次测试失败
    def _fail_call(*args, **kwargs):
        raise AssertionError("LLM 抽取不应被调用——占位 + 无 desc 应当直接跳过")

    monkeypatch.setattr(extractor_mod, "_llm_extract_file", _fail_call)

    result = asyncio.run(
        extractor_mod.extract_from_file(
            user_id="u-1",
            role_id="default",
            file_name="拉师傅来我家的三年.mp3.mpeg",
            modality="audio",
            desc="",  # 用户没填描述
            parsed_content="[音频转录失败] 拉师傅来我家的三年.mp3.mpeg",
            source_meta={"fileId": "f-1"},
        )
    )

    assert result == {"count": 0, "inserted_ids": [], "summary": ""}, (
        f"占位 + 无 desc 应返回空结果，实际 {result}"
    )


def _make_async_return(value):
    """构造一个返回固定值的 async 函数。"""
    async def _fn(*args, **kwargs):
        return value
    return _fn


def test_extract_proceeds_when_placeholder_but_desc_provided(monkeypatch):
    """回归测试：转录失败占位 + 有 desc → extract_from_file 仍调用 LLM（仅依赖 desc）。"""
    from memory import extractor as extractor_mod

    captured: dict = {}

    async def _fake_llm_extract(file_name, modality, desc, parsed_content):
        captured["file_name"] = file_name
        captured["modality"] = modality
        captured["desc"] = desc
        captured["parsed_content"] = parsed_content
        return [{"fact": "用户提供了描述", "causes": "", "level": "L0", "emotion": "neutral", "intensity": 0.0, "relation": ""}]

    monkeypatch.setattr(extractor_mod, "_llm_extract_file", _fake_llm_extract)
    monkeypatch.setattr(extractor_mod, "_vectorize_and_store", _make_async_return([]))
    monkeypatch.setattr(extractor_mod, "_summarize", _make_async_return(""))
    # _archive 返回 1 个插入 id，对应 count=1
    monkeypatch.setattr(extractor_mod, "_archive", _make_async_return([101]))
    monkeypatch.setattr(extractor_mod, "_dedup_one", _make_async_return({"duplicate": False, "merged": "用户提供了描述", "relation": ""}))

    result = asyncio.run(
        extractor_mod.extract_from_file(
            user_id="u-1",
            role_id="default",
            file_name="voice.mp3",
            modality="audio",
            desc="这是拉师傅的日常",
            parsed_content="[音频转录失败] voice.mp3",
            source_meta={"fileId": "f-1"},
        )
    )

    assert captured.get("desc") == "这是拉师傅的日常"
    assert "音频转录失败" in captured.get("parsed_content", "")
    assert result["count"] >= 1, f"有 desc 时应生成记忆，实际 {result}"


def test_extract_proceeds_when_real_content(monkeypatch):
    """回归测试：正常内容 → extract_from_file 正常调用 LLM（未被本次修改误伤）。"""
    from memory import extractor as extractor_mod

    called: dict = {"flag": False}

    async def _fake_llm_extract(file_name, modality, desc, parsed_content):
        called["flag"] = True
        return [{"fact": "测试记忆", "causes": "", "level": "L1", "emotion": "neutral", "intensity": 0.0, "relation": ""}]

    monkeypatch.setattr(extractor_mod, "_llm_extract_file", _fake_llm_extract)
    monkeypatch.setattr(extractor_mod, "_vectorize_and_store", _make_async_return([]))
    monkeypatch.setattr(extractor_mod, "_summarize", _make_async_return(""))
    monkeypatch.setattr(extractor_mod, "_archive", _make_async_return([102]))
    monkeypatch.setattr(extractor_mod, "_dedup_one", _make_async_return({"duplicate": False, "merged": "测试记忆", "relation": ""}))

    asyncio.run(
        extractor_mod.extract_from_file(
            user_id="u-1",
            role_id="default",
            file_name="voice.mp3",
            modality="audio",
            desc="",
            parsed_content="你好世界，这是真实的转录文本。",
            source_meta={"fileId": "f-1"},
        )
    )

    assert called["flag"] is True, "正常内容场景不应被跳过"


def test_placeholder_detection_helper():
    """_is_placeholder_parsed_content 辅助函数本身覆盖所有已知占位前缀。"""
    from memory.extractor import _is_placeholder_parsed_content

    # 占位应判定为 True
    assert _is_placeholder_parsed_content("[音频转录失败] voice.mp3") is True
    assert _is_placeholder_parsed_content("[视频关键帧抽取失败] clip.mp4") is True
    assert _is_placeholder_parsed_content("[图片描述失败] photo.jpg") is True
    assert _is_placeholder_parsed_content("[文档解析失败] report.pdf") is True

    # 真实内容应判定为 False
    assert _is_placeholder_parsed_content("你好世界") is False
    assert _is_placeholder_parsed_content("今天天气不错") is False

    # 边界：空字符串、纯空白
    assert _is_placeholder_parsed_content("") is False
    assert _is_placeholder_parsed_content("   ") is False
