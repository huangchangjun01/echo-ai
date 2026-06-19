"""Integration-style tests for the FastAPI app using TestClient."""
from __future__ import annotations

import sys
import types

import pytest
from fastapi.testclient import TestClient


def _patch_embed_module():
    fake = types.ModuleType("embedding.models")

    def compute_text_embeddings(texts: list[str], device: str = "auto"):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def compute_image_embeddings(images, device: str = "auto"):
        return [[0.1, 0.2, 0.3, 0.4] for _ in images]

    fake.compute_text_embeddings = compute_text_embeddings
    fake.compute_image_embeddings = compute_image_embeddings
    fake.compute_embedding = lambda b: [0.1, 0.2, 0.3, 0.4]
    fake.compute_text_embedding = lambda t: compute_text_embeddings([t])[0]
    fake.load_model = lambda device="auto": (None, None, None)
    fake.warmup = lambda device="auto", batch_size=1: None
    fake.detect_device = lambda preferred="auto": "cpu"
    sys.modules["embedding.models"] = fake


def _patch_vector_store(monkeypatch):
    from vector import vector_store as vs_mod

    class FakeVS:
        def __init__(self):
            self.threshold = 0.5
            self.last_where = None
            self.calls = 0

        def query(self, query_text, n_results=5, embedding_fn=None, where=None):
            self.calls += 1
            self.last_where = where
            tenant = (where or {}).get("userId", "u1")
            return {
                "ids": [["uuid-1"]],
                "documents": [["some doc for " + tenant]],
                "metadatas": [[{"fileId": "f1", "userId": tenant, "similarity": 0.9}]],
                "distances": [[0.1]],
            }

        def add_texts(self, **kw):
            pass

        @property
        def _collection(self):
            class _C:
                def exists(self_inner):
                    return True

            return _C()

    fake = FakeVS()
    monkeypatch.setattr(vs_mod, "get_vector_store", lambda: fake)
    return fake


_FAKE_VS = None


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    _patch_embed_module()
    global _FAKE_VS
    _FAKE_VS = _patch_vector_store(monkeypatch)
    yield


def test_chat_requires_user_id_for_tenant_isolation():
    from app.agent_runner import app

    with TestClient(app) as client:
        resp = client.post("/chat", json={"query": "hello", "k": 3})  # no userId
    assert resp.status_code == 422
    assert "userId" in resp.text


def test_chat_returns_candidates_with_real_uuid():
    from app.agent_runner import app

    with TestClient(app) as client:
        resp = client.post("/chat", json={"query": "hello", "k": 3, "userId": "u1"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["query"] == "hello"
    assert data["userId"] == "u1"
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["id"] == "uuid-1"
    assert data["candidates"][0]["metadata"]["similarity"] == 0.9


def test_chat_passes_userid_to_vectorstore():
    from app.agent_runner import app

    with TestClient(app) as client:
        client.post("/chat", json={"query": "hello", "k": 3, "userId": "u42"})
    assert _FAKE_VS.last_where == {"userId": "u42"}, (
        "API must inject userId into the vector store filter"
    )


def test_chat_rejects_blank_user_id():
    from app.agent_runner import app

    with TestClient(app) as client:
        resp = client.post("/chat", json={"query": "hello", "k": 3, "userId": ""})
    assert resp.status_code == 422


def test_chat_validates_query_required():
    from app.agent_runner import app

    with TestClient(app) as client:
        resp = client.post("/chat", json={"k": 5, "userId": "u1"})  # missing query
    assert resp.status_code == 422


def test_health_endpoint_returns_ok():
    from app.agent_runner import app

    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}