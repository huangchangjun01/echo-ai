"""端到端 HTTP 测试：验证 /ingest_file 接口在音频转录失败时的记忆兜底逻辑。

绕过 SSRF 限制：通过 monkeypatch download_file_async 直接返回测试音频字节。
"""
from __future__ import annotations

import asyncio
import json
import sys
import types

from fastapi.testclient import TestClient


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


def _patch_whisper(monkeypatch):
    """Mock whisper 让转录返回空（模拟失败）"""
    from embedding import whisper as whisper_mod

    def fake_embed_audio(audio_bytes: bytes) -> dict:
        return {"text": "", "embedding": [0.0] * 4, "dim": 4}

    monkeypatch.setattr(whisper_mod, "embed_audio", fake_embed_audio)


def _patch_vector_store():
    """Mock vector store，捕获所有 add_texts 调用"""
    from vector import vector_store as vs_mod

    class FakeVS:
        def __init__(self):
            self.calls = []

        def query(self, *args, **kwargs):
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        def add_texts(self, ids=None, texts=None, metadatas=None, embeddings=None):
            self.calls.append({
                "ids": list(ids or []),
                "texts": list(texts or []),
                "metadatas": list(metadatas or []),
            })

        @property
        def _collection(self):
            class _C:
                def exists(self_inner):
                    return True
            return _C()

    fake = FakeVS()
    vs_module_get = lambda: fake
    import vector.vector_store
    vector.vector_store.get_vector_store = vs_module_get
    return fake


def _patch_download(monkeypatch):
    """Mock download_file_async 返回 MP3 字节"""
    from biz import ingest as ingest_mod

    async def fake_download(url, **kw):
        return b"\xff\xfb\x90\x00" + b"\x00" * 4096

    monkeypatch.setattr(ingest_mod, "download_file_async", fake_download)


def test_audio_failure_via_http_endpoint(monkeypatch):
    """通过 FastAPI TestClient 触发 /ingest_file，验证音频转录失败兜底逻辑。"""
    _patch_embed_module()
    _patch_whisper(monkeypatch)
    _patch_download(monkeypatch)
    fake_vs = _patch_vector_store()

    from app.agent_runner import app

    payload = {
        "userId": "e2e-http-user",
        "file": {
            "fileId": "audio-http-1",
            "fileName": "test_fake_audio.mp3",
            "fileKey": "",
            "url": "http://example.com/fake.mp3",
        },
        "desc": "测试音频转录失败的兜底逻辑",
        "roleId": "default",
    }

    with TestClient(app) as client:
        # 调用 /ingest_file（异步后台任务）
        r = client.post("/ingest_file", json=payload)
        assert r.status_code == 200, f"/ingest_file 返回 {r.status_code}: {r.text}"
        data = r.json()
        print(f"[http] /ingest_file 响应: {data}")
        assert data.get("ok") is True
        assert data.get("queued") is True

        # TestClient 会在 with 退出时自动等待后台任务完成
        # 重新发起请求让 TestClient 处理 BackgroundTasks

    # 验证 vector store 收到了一次 add_texts 调用
    print(f"[http] add_texts 调用次数: {len(fake_vs.calls)}")
    assert len(fake_vs.calls) >= 1, "音频入库应触发至少一次 add_texts"

    # 找到音频那次的调用（modality=audio）
    audio_calls = [c for c in fake_vs.calls if c["metadatas"] and c["metadatas"][0].get("modality") == "audio"]
    assert len(audio_calls) == 1, f"应恰好一次音频 add_texts，实际 {len(audio_calls)}"

    call = audio_calls[0]
    meta = call["metadatas"][0]
    text = call["texts"][0]

    print(f"[http] audio add_texts:")
    print(f"  ids: {call['ids']}")
    print(f"  text: {text[:80]}")
    print(f"  metadata.transcribeStatus: {meta.get('transcribeStatus')}")
    print(f"  metadata.modality: {meta.get('modality')}")

    # 关键断言 1：转录失败仍写入 EchoDoc
    assert call["ids"] == ["audio-http-1"]
    # 关键断言 2：占位文本被写入
    assert "音频转录失败" in text, f"占位文本未写入，实际: {text}"
    # 关键断言 3：metadata 标记 transcribeStatus="failed"
    assert meta.get("transcribeStatus") == "failed"
    assert meta.get("modality") == "audio"
    print("[http] ALL CHECKS PASSED")


if __name__ == "__main__":
    # 当作脚本运行时的入口
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
