"""Unit tests for ingest pipeline."""
from __future__ import annotations

import asyncio
import sys
import types

import pytest


def _patch_embed_module(monkeypatch, *, text_fn=None, image_fn=None):
    fake = types.ModuleType("embedding.models")
    fake.compute_text_embeddings = text_fn or (lambda texts, device="auto": [[0.1] * 4 for _ in texts])
    fake.compute_image_embeddings = image_fn or (lambda images, device="auto": [[0.2] * 4 for _ in images])
    fake.compute_embedding = lambda b: [0.1, 0.2, 0.3, 0.4]
    fake.compute_text_embedding = lambda t: (text_fn or (lambda texts, device="auto": [[0.1] * 4 for _ in texts]))([t])[0]
    fake.load_model = lambda device="auto": (None, None, None)
    fake.warmup = lambda device="auto", batch_size=1: None
    fake.detect_device = lambda preferred="auto": "cpu"
    sys.modules["embedding.models"] = fake
    return fake


@pytest.fixture(autouse=True)
def _fake_models(monkeypatch):
    _patch_embed_module(monkeypatch)
    yield


def test_ingest_text_file_chunks_and_persists(monkeypatch):
    from biz import ingest as ingest_mod
    from embedding.embeddings import ChineseCLIPEmbeddings

    async def fake_download(url: str) -> bytes:
        return ("这是中文测试文本。" * 200).encode("utf-8")

    monkeypatch.setattr(ingest_mod, "download_file_async", fake_download)

    store = types.SimpleNamespace()

    persisted = {}

    def add_texts(ids, texts, metadatas=None, embeddings=None):
        persisted["ids"] = list(ids)
        persisted["texts"] = list(texts)
        persisted["metadatas"] = list(metadatas or [])
        persisted["embeddings"] = list(embeddings or [])

    store.add_texts = add_texts

    embeddings = ChineseCLIPEmbeddings()
    file_obj = {"fileId": "f1", "fileName": "a.txt", "fileKey": "a.txt"}
    result = asyncio.run(ingest_mod.ingest_file("user-1", file_obj, embeddings, store))

    assert result.success, result.error
    assert result.chunks >= 1
    assert all(meta["userId"] == "user-1" for meta in persisted["metadatas"])
    assert all(meta["fileId"] == "f1" for meta in persisted["metadatas"])
    assert persisted["ids"][0].startswith("f1:")


def test_ingest_image_file(monkeypatch):
    from biz import ingest as ingest_mod
    from embedding.embeddings import ChineseCLIPEmbeddings

    async def fake_download(url: str) -> bytes:
        # PNG signature
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

    monkeypatch.setattr(ingest_mod, "download_file_async", fake_download)

    persisted = {}

    def add_texts(ids, texts, metadatas=None, embeddings=None):
        persisted["ids"] = list(ids)

    store = types.SimpleNamespace(add_texts=add_texts)
    embeddings = ChineseCLIPEmbeddings()
    file_obj = {"fileId": "img-1", "fileName": "x.png", "url": "http://example.com/x.png"}
    result = asyncio.run(ingest_mod.ingest_file("u", file_obj, embeddings, store))
    assert result.success, result.error
    assert persisted["ids"] == ["img-1"]


def test_ingest_unsupported_mime_rejected(monkeypatch):
    from biz import ingest as ingest_mod
    from embedding.embeddings import ChineseCLIPEmbeddings

    async def fake_download(url: str) -> bytes:
        # MP4 magic bytes
        return b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 32

    monkeypatch.setattr(ingest_mod, "download_file_async", fake_download)

    store = types.SimpleNamespace(add_texts=lambda **kw: None)
    embeddings = ChineseCLIPEmbeddings()
    file_obj = {"fileId": "v1", "fileName": "v.mp4", "url": "http://example.com/v.mp4"}
    result = asyncio.run(ingest_mod.ingest_file("u", file_obj, embeddings, store))
    assert not result.success
    assert "Unsupported" in (result.error or "")


def test_ingest_download_failure(monkeypatch):
    from biz import ingest as ingest_mod
    from embedding.embeddings import ChineseCLIPEmbeddings
    from utils.downloader import DownloadError

    async def fake_download(url: str) -> bytes:
        raise DownloadError("blocked")

    monkeypatch.setattr(ingest_mod, "download_file_async", fake_download)
    store = types.SimpleNamespace(add_texts=lambda **kw: None)
    embeddings = ChineseCLIPEmbeddings()
    file_obj = {"fileId": "x", "fileName": "x.txt", "url": "http://example.com/x.txt"}
    result = asyncio.run(ingest_mod.ingest_file("u", file_obj, embeddings, store))
    assert not result.success
    assert "Download failed" in (result.error or "")


def test_ingest_audio_transcribe_failure_falls_back_to_placeholder(monkeypatch):
    """回归测试：音频转录失败时仍返回 success，且写入占位 EchoDoc，
    让 _maybe_generate_memory 能从 desc 兜底生成记忆条目。

    修复前的 bug：转录失败时 result.success=False → _maybe_generate_memory 直接
    return → 用户上传音频后没有任何记忆条目。
    """
    from biz import ingest as ingest_mod
    from embedding import whisper as whisper_mod
    from embedding.embeddings import ChineseCLIPEmbeddings

    # Mock download：MP3 帧头（含 0xFF 同步字节 + MPEG 标识）确保 _detect_mime 识别为 audio/mpeg
    # 否则纯文本字节会被兜底检测成 text/plain 走 _ingest_text 分支。
    async def fake_download(url: str) -> bytes:
        return b"\xff\xfb\x90\x00" + b"\x00" * 64

    monkeypatch.setattr(ingest_mod, "download_file_async", fake_download)

    # Mock whisper 让转录返回空文本（模拟转录失败）
    def fake_embed_audio(audio_bytes: bytes) -> dict:
        return {"text": "", "embedding": [0.0] * 4, "dim": 4}

    monkeypatch.setattr(whisper_mod, "embed_audio", fake_embed_audio)

    # 收集 add_texts 调用
    persisted: dict = {}

    def add_texts(ids, texts, metadatas=None, embeddings=None):
        persisted["ids"] = list(ids)
        persisted["texts"] = list(texts)
        persisted["metadatas"] = list(metadatas or [])

    store = types.SimpleNamespace(add_texts=add_texts)
    embeddings = ChineseCLIPEmbeddings()
    file_obj = {
        "fileId": "audio-1",
        "fileName": "voice.mp3",
        "fileKey": "voice.mp3",
        "url": "http://example.com/voice.mp3",
    }

    result = asyncio.run(ingest_mod.ingest_file("u", file_obj, embeddings, store))

    # 关键断言 1：转录失败仍 success（不再 hard-fail），让 _maybe_generate_memory 走 desc 兜底
    assert result.success, f"音频转录失败应返回 success，实际 error={result.error}"
    # 关键断言 2：占位文本被写入 EchoDoc（用户能在记忆管理列表中看到）
    assert persisted.get("ids") == ["audio-1"]
    assert "音频转录失败" in persisted["texts"][0]
    assert persisted["metadatas"][0]["transcribeStatus"] == "failed"
    # 关键断言 3：parsed_text 非空，让 _maybe_generate_memory 能走 desc 兜底
    assert "音频转录失败" in (result.parsed_text or "")
    assert result.modality == "audio"


def test_ingest_audio_transcribe_success_unchanged(monkeypatch):
    """回归测试：音频转录成功时行为不变（未被本次修改破坏）。"""
    from biz import ingest as ingest_mod
    from embedding import whisper as whisper_mod
    from embedding.embeddings import ChineseCLIPEmbeddings

    async def fake_download(url: str) -> bytes:
        return b"\xff\xfb\x90\x00" + b"\x00" * 64

    monkeypatch.setattr(ingest_mod, "download_file_async", fake_download)

    # Mock whisper 让转录正常返回
    def fake_embed_audio(audio_bytes: bytes) -> dict:
        return {"text": "你好世界，这是正常的转录文本。", "embedding": [0.1] * 4, "dim": 4}

    monkeypatch.setattr(whisper_mod, "embed_audio", fake_embed_audio)

    persisted: dict = {}

    def add_texts(ids, texts, metadatas=None, embeddings=None):
        persisted["ids"] = list(ids)
        persisted["texts"] = list(texts)
        persisted["metadatas"] = list(metadatas or [])

    store = types.SimpleNamespace(add_texts=add_texts)
    embeddings = ChineseCLIPEmbeddings()
    file_obj = {
        "fileId": "audio-ok-1",
        "fileName": "good_audio.mp3",
        "fileKey": "good_audio.mp3",
        "url": "http://example.com/good.mp3",
    }

    result = asyncio.run(ingest_mod.ingest_file("u", file_obj, embeddings, store))

    assert result.success
    # 关键：转录成功时不写占位文本，写的是真实转录文本
    assert persisted["ids"] == ["audio-ok-1"]
    assert persisted["texts"][0] == "你好世界，这是正常的转录文本。"
    assert "transcribeStatus" not in persisted["metadatas"][0]
    assert result.parsed_text == "你好世界，这是正常的转录文本。"
    assert result.modality == "audio"