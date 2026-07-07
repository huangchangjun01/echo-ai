"""记忆向量库：使用 BGE-M3 (768 维) 存储长期记忆/对话片段。

与 `vector_store.py` 分离是因为维度不同（512 vs 768），不能复用同一个 collection。
接口与 `WeaviateVectorStore` 保持一致，便于上层透明使用。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Sequence
from typing import Any

from config.config import get_settings
from utils.request_context import merge_extra

from .vector_store import _WeaviateHttpClient

logger = logging.getLogger(__name__)


class MemoryVectorStore:
    """基于 Weaviate 的记忆向量库（BGE-M3 / 768 维）。"""

    def __init__(self) -> None:
        settings = get_settings()
        cfg = settings.weaviate
        self.class_name: str = cfg.memory_class
        # EchoMemory 存 BGE-M3 768 维，文本相似度普遍 0.5~0.95，0.7 合理。
        self.threshold: float = (
            settings.weaviate.memory_threshold
            if getattr(settings.weaviate, "memory_threshold", None) is not None
            else settings.vector_similarity_threshold
        )
        self._http = _WeaviateHttpClient.from_settings()
        logger.info("MemoryVectorStore at %s class=%s", cfg.resolved_url(), self.class_name)
        self._ensure_collection()

    _EXPECTED_PROPERTIES: tuple[dict, ...] = (
        {"name": "text", "dataType": ["text"]},
        {"name": "metadata", "dataType": ["text"]},
        {"name": "userId", "dataType": ["text"], "indexFilterable": True},
        {"name": "memoryId", "dataType": ["int"]},
        {"name": "level", "dataType": ["text"]},
        {"name": "emotion", "dataType": ["text"]},
    )

    def _ensure_collection(self) -> None:
        if self._http.class_exists(self.class_name):
            self._ensure_filterable_properties()
            return
        logger.info("Creating memory collection %s", self.class_name)
        class_obj = {
            "class": self.class_name,
            "vectorizer": "none",
            "properties": [dict(p) for p in self._EXPECTED_PROPERTIES],
        }
        self._http.create_class(class_obj)

    def _ensure_filterable_properties(self) -> None:
        cfg = self._http.get_class(self.class_name)
        if not cfg:
            return
        existing = {p.get("name") for p in cfg.get("properties", []) or []}
        for prop in self._EXPECTED_PROPERTIES:
            if prop["name"] in existing:
                continue
            try:
                if self._http.add_property(self.class_name, prop):
                    logger.info("Added property %s to %s", prop["name"], self.class_name)
            except Exception as e:
                logger.warning("Failed to add property %s: %s", prop["name"], e)

    def add_texts(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict | None] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        if embeddings is None:
            raise ValueError("MemoryVectorStore requires embeddings")
        if len(ids) != len(texts) or len(ids) != len(embeddings):
            raise ValueError("ids/texts/embeddings must have equal length")
        if metadatas is None:
            metadatas = [None] * len(texts)

        from .vector_store import _ensure_uuid

        objects: list[dict] = []
        returned_ids: list[str] = []
        for id_, text, md, vec in zip(ids, texts, metadatas, embeddings):
            md = md or {}
            uuid = _ensure_uuid(id_)
            obj = {
                "id": uuid,
                "class": self.class_name,
                "properties": {
                    "text": text,
                    "metadata": json.dumps(md, ensure_ascii=False),
                    "userId": md.get("userId", md.get("user_id", "")),
                    "memoryId": int(md.get("memoryId", md.get("memory_id", 0)) or 0),
                    "level": md.get("level", "L1"),
                    "emotion": md.get("emotion", "neutral"),
                },
                "vector": list(map(float, vec)),
            }
            objects.append(obj)
            returned_ids.append(uuid)

        t0 = time.perf_counter()
        result = self._http.batch_insert(objects)
        insert_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "memory insert",
            extra=merge_extra(
                stage="memory_vector_add",
                event="ok",
                class_name=self.class_name,
                count=len(result or returned_ids),
                http_ms=insert_ms,
            ),
        )
        return result or returned_ids

    def query(
        self,
        query_text: str,
        n_results: int = 8,
        embedding_fn: Any | None = None,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        t_total = time.perf_counter()
        if embedding_fn is None:
            raise ValueError("embedding_fn required")
        if not query_text:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        tenant_id = (where or {}).get("userId") or (where or {}).get("user_id")
        if not tenant_id:
            raise ValueError("where.userId required")

        try:
            t0 = time.perf_counter()
            q_emb = embedding_fn([query_text])[0]
            embed_ms = round((time.perf_counter() - t0) * 1000, 2)
        except Exception as e:
            logger.error(
                "memory embedding_fn failed",
                extra=merge_extra(
                    stage="memory_vector_query",
                    event="embed_error",
                    tenant=tenant_id,
                    error=str(e)[:200],
                ),
            )
            raise ValueError("embedding_fn failed") from e

        from .vector_store import (
            _build_search_query,
            _extract_get_payload,
        )

        fetch_limit = max(int(n_results) * 10, 20)
        # EchoMemory has no fileId/fileName/chunkIndex — only memory-specific fields.
        memory_fields = (
            "text metadata userId memoryId level emotion _additional { id distance }"
        )
        query = _build_search_query(
            self.class_name, list(q_emb), fetch_limit, where, fields=memory_fields
        )
        try:
            t1 = time.perf_counter()
            payload = self._http.graphql(query)
            http_ms = round((time.perf_counter() - t1) * 1000, 2)
            items = _extract_get_payload(payload, self.class_name)
        except Exception as e:
            logger.error(
                "memory query failed",
                extra=merge_extra(
                    stage="memory_vector_query",
                    event="http_error",
                    tenant=tenant_id,
                    error=str(e)[:300],
                ),
            )
            raise

        scored: list[tuple[float, str, dict]] = []
        for item in items:
            additional = item.get("_additional") or {}
            distance = additional.get("distance")
            if distance is None:
                continue
            similarity = 1.0 - float(distance)
            if similarity >= self.threshold:
                scored.append((similarity, str(additional.get("id") or item.get("id") or ""), item))

        scored.sort(key=lambda x: x[0], reverse=True)
        raw_hits = len(scored)
        scored = scored[: int(n_results)]
        score_dist: list[float] = []

        ids, docs, mds, distances = [], [], [], []
        for similarity, obj_id, item in scored:
            ids.append(obj_id)
            docs.append(item.get("text", ""))
            try:
                meta = json.loads(item.get("metadata") or "{}")
            except Exception:
                meta = {}
            meta["similarity"] = round(similarity, 4)
            mds.append(meta)
            distances.append(round(1.0 - similarity, 6))
            score_dist.append(round(similarity, 4))

        top_score = max(score_dist) if score_dist else 0.0
        median_score = sorted(score_dist)[len(score_dist) // 2] if score_dist else 0.0
        logger.info(
            "memory vector query",
            extra=merge_extra(
                stage="memory_vector_query",
                event="ok",
                class_name=self.class_name,
                tenant=tenant_id,
                n_results=n_results,
                fetch_limit=fetch_limit,
                threshold=self.threshold,
                raw_hits=raw_hits,
                final_hits=len(ids),
                top_score=round(top_score, 4),
                median_score=round(median_score, 4),
                embed_ms=embed_ms,
                http_ms=http_ms,
                duration_ms=round((time.perf_counter() - t_total) * 1000, 2),
                query_preview=(query_text or "")[:80],
            ),
        )

        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [mds],
            "distances": [distances],
        }

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass


_singleton: MemoryVectorStore | None = None
_singleton_lock = threading.Lock()


def get_memory_vector_store() -> MemoryVectorStore:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = MemoryVectorStore()
    return _singleton


def reset_memory_vector_store() -> None:
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            try:
                _singleton.close()
            except Exception:
                pass
        _singleton = None