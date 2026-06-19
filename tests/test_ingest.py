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