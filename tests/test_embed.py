"""Unit tests for the ChineseCLIPEmbeddings wrapper (no real model loading)."""
from __future__ import annotations

import sys
import types

import pytest


def _install_fake_models():
    """Patch `embedding.models` so embeddings tests do not load real weights."""
    fake = types.ModuleType("embedding.models")

    def compute_text_embeddings(texts: list[str], device: str = "auto"):
        # Deterministic, normalized 4-dim vector per text.
        out = []
        for t in texts:
            v = [float((ord(c) % 7) - 3) / 3.0 for c in (t or "x")[:4]]
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out

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
    return fake


@pytest.fixture(autouse=True)
def _fake_models(monkeypatch):
    _install_fake_models()
    yield


def test_embed_documents_returns_list_of_vectors():
    from embedding.embeddings import ChineseCLIPEmbeddings

    emb = ChineseCLIPEmbeddings()
    vecs = emb.embed_documents(["hello", "world"])
    assert len(vecs) == 2
    assert all(isinstance(v, list) for v in vecs)
    assert all(all(isinstance(x, float) for x in v) for v in vecs)


def test_embed_query_single_vector():
    from embedding.embeddings import ChineseCLIPEmbeddings

    emb = ChineseCLIPEmbeddings()
    v = emb.embed_query("hi")
    assert isinstance(v, list)
    assert v


def test_embed_images_returns_list_of_vectors():
    from embedding.embeddings import ChineseCLIPEmbeddings

    emb = ChineseCLIPEmbeddings()
    vecs = emb.embed_images([b"\x89PNG\r\n\x1a\n" + b"x" * 16, b"\xff\xd8\xff" + b"x" * 16])
    assert len(vecs) == 2