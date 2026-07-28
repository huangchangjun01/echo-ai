"""回忆记忆向量库：EchoRecall 集合（独立于 EchoDoc / EchoMemory）。

设计要点：
- 只索引"摘要"：外部只看到摘要 → 命中后按需从对象存储拉 md 细节（渐进式回忆）。
- 一条 memoryId 最多一条记录（upsert 语义，主题更新/源文件追加时覆盖）。
- 隔离维度：BGE-M3 + (userId, roleId, memoryId)。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid as _uuid
from collections.abc import Sequence
from typing import Any

from config.config import get_settings
from utils.request_context import log_exception, merge_extra

from .vector_store import (
    _WeaviateHttpClient,
    _build_search_query,
    _ensure_uuid,
    _extract_get_payload,
)

logger = logging.getLogger(__name__)


class RecallVectorStore:
    """EchoRecall：回忆记忆的摘要向量库（BGE-M3）。"""

    def __init__(self) -> None:
        cfg = get_settings().weaviate
        self.class_name: str = cfg.recall_class
        # 摘要相似阈值：参考 memory_threshold；可按需调
        self.threshold: float = (
            cfg.recall_threshold
            if getattr(cfg, "recall_threshold", None) is not None
            else cfg.memory_threshold
        )
        self._http = _WeaviateHttpClient.from_settings()
        logger.info("RecallVectorStore at %s class=%s", cfg.resolved_url(), self.class_name)
        self._ensure_collection()

    _EXPECTED_PROPERTIES: tuple[dict, ...] = (
        {"name": "summary", "dataType": ["text"]},
        {"name": "metadata", "dataType": ["text"]},
        {"name": "userId", "dataType": ["text"], "indexFilterable": True},
        {"name": "roleId", "dataType": ["text"], "indexFilterable": True},
        {"name": "memoryId", "dataType": ["text"], "indexFilterable": True},
        {"name": "mdKey", "dataType": ["text"]},
        {"name": "topic", "dataType": ["text"]},
        {"name": "intensity", "dataType": ["number"]},
    )

    def _ensure_collection(self) -> None:
        if self._http.class_exists(self.class_name):
            self._ensure_filterable_properties()
            return
        logger.info("Creating recall collection %s", self.class_name)
        self._http.create_class(
            {"class": self.class_name, "vectorizer": "none", "properties": [dict(p) for p in self._EXPECTED_PROPERTIES]}
        )

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
                log_exception(
                    logger,
                    "add_property recall failed (skipped)",
                    exc=e,
                    level=logging.WARNING,
                    stage="recall_schema",
                    event="add_prop_error",
                    class_name=self.class_name,
                    prop_name=prop["name"],
                )

    # ---------- write ----------

    def upsert(
        self,
        memory_id: str,
        summary: str,
        *,
        user_id: str,
        role_id: str,
        md_key: str,
        topic: str,
        embedding: Sequence[float],
        intensity: float | None = None,
    ) -> str:
        """单条 upsert：先按 memoryId 查已存在则删后插；不存在直接插。

        内部 ID 用稳定 hash，避免 _ensure_uuid 派新 UUID 阻断同主题 upsert。
        """
        stable_id = _stable_uuid(f"{user_id}:{memory_id}")
        # 先尝试删旧记录（幂等）
        try:
            self._http.delete_object(self.class_name, stable_id)
        except Exception as e:
            log_exception(
                logger,
                "recall upsert: pre-delete failed (ignore)",
                exc=e,
                level=logging.DEBUG,
                stage="recall_write",
                event="pre_delete_error",
                memory_id=memory_id,
            )

        md: dict[str, Any] = {
            "userId": user_id,
            "roleId": role_id or "default",
            "memoryId": memory_id,
            "mdKey": md_key,
            "topic": topic,
        }
        if intensity is not None:
            md["intensity"] = float(intensity)

        obj: dict[str, Any] = {
            "id": stable_id,
            "class": self.class_name,
            "properties": {
                "summary": summary,
                "metadata": json.dumps(md, ensure_ascii=False),
                "userId": user_id,
                "roleId": role_id or "default",
                "memoryId": memory_id,
                "mdKey": md_key,
                "topic": topic,
                "intensity": float(intensity) if intensity is not None else 0.0,
            },
            "vector": list(map(float, embedding)),
        }
        t0 = time.perf_counter()
        try:
            self._http.batch_insert([obj])
        except Exception as e:
            log_exception(
                logger,
                "recall upsert batch_insert failed",
                exc=e,
                level=logging.ERROR,
                stage="recall_write",
                event="insert_error",
                memory_id=memory_id,
            )
            raise
        logger.info(
            "recall upsert ok",
            extra=merge_extra(
                stage="recall_write",
                event="ok",
                class_name=self.class_name,
                memory_id=memory_id,
                http_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return stable_id

    def delete_by_memory_id(self, user_id: str, memory_id: str) -> int:
        """按 (userId, memoryId) 删。返回受影响条数（实际为 0/1）。"""
        stable_id = _stable_uuid(f"{user_id}:{memory_id}")
        try:
            self._http.delete_object(self.class_name, stable_id)
            logger.info(
                "recall delete ok",
                extra=merge_extra(stage="recall_write", event="ok", memory_id=memory_id),
            )
            return 1
        except Exception as e:
            log_exception(
                logger,
                "recall delete failed",
                exc=e,
                level=logging.WARNING,
                stage="recall_write",
                event="delete_error",
                memory_id=memory_id,
            )
            return 0

    # ---------- read ----------

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        embedding_fn: Any | None = None,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Top-k 摘要检索。where 必须含 userId（多租户隔离）。"""
        t0 = time.perf_counter()
        if embedding_fn is None:
            raise ValueError("embedding_fn required")
        tenant_id = (where or {}).get("userId")
        if not tenant_id:
            raise ValueError("where.userId required for tenant isolation")

        try:
            q_emb = embedding_fn([query_text])[0]
        except Exception as e:
            log_exception(
                logger,
                "recall query embedding failed",
                exc=e,
                level=logging.WARNING,
                stage="recall_query",
                event="embed_error",
                tenant=tenant_id,
            )
            raise

        fetch_limit = max(int(n_results) * 10, 20)
        fields = "summary metadata userId roleId memoryId mdKey topic _additional { id distance }"
        q = _build_search_query(self.class_name, list(q_emb), fetch_limit, where, fields=fields)
        try:
            payload = self._http.graphql(q)
            items = _extract_get_payload(payload, self.class_name)
        except Exception as e:
            log_exception(
                logger,
                "recall query failed",
                exc=e,
                level=logging.ERROR,
                stage="recall_query",
                event="http_error",
                tenant=tenant_id,
            )
            raise

        scored: list[tuple[float, str, dict]] = []
        for item in items:
            add = item.get("_additional") or {}
            d = add.get("distance")
            if d is None:
                continue
            sim = 1.0 - float(d)
            if sim >= self.threshold:
                scored.append((sim, str(add.get("id") or item.get("id") or ""), item))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[: int(n_results)]

        ids, summaries, mds, distances = [], [], [], []
        for sim, obj_id, item in scored:
            ids.append(obj_id)
            summaries.append(item.get("summary", ""))
            md_raw = item.get("metadata") or "{}"
            try:
                md = json.loads(md_raw)
            except Exception:
                md = {}
            md["similarity"] = round(sim, 4)
            mds.append(md)
            distances.append(round(1.0 - sim, 6))

        logger.info(
            "recall query",
            extra=merge_extra(
                stage="recall_query",
                event="ok",
                class_name=self.class_name,
                tenant=tenant_id,
                n_results=n_results,
                final_hits=len(ids),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return {"ids": [ids], "summaries": [summaries], "metadatas": [mds], "distances": [distances]}

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def _query_with_vector(
        self,
        vector: Sequence[float],
        *,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """直接接受已计算好的 query 向量，跳过 query() 内部 embed。

        返回元素：{id, summary, _meta, similarity}
        """
        tenant_id = (where or {}).get("userId")
        if not tenant_id:
            raise ValueError("where.userId required for tenant isolation")
        fetch_limit = max(int(n_results) * 10, 20)
        fields = "summary metadata userId roleId memoryId mdKey topic _additional { id distance }"
        q = _build_search_query(self.class_name, list(vector), fetch_limit, where, fields=fields)
        payload = self._http.graphql(q)
        items = _extract_get_payload(payload, self.class_name)
        scored: list[dict[str, Any]] = []
        for it in items:
            add = it.get("_additional") or {}
            d = add.get("distance")
            if d is None:
                continue
            sim = 1.0 - float(d)
            if sim < self.threshold:
                continue
            md_raw = it.get("metadata") or "{}"
            try:
                meta = json.loads(md_raw)
            except Exception:
                meta = {}
            scored.append(
                {
                    "id": str(add.get("id") or it.get("id") or ""),
                    "summary": it.get("summary", ""),
                    "_meta": meta,
                    "similarity": round(sim, 4),
                }
            )
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[: int(n_results)]


_singleton: RecallVectorStore | None = None
_lock = threading.Lock()


def get_recall_store() -> RecallVectorStore:
    global _singleton
    if _singleton is None:
        with _lock:
            if _singleton is None:
                _singleton = RecallVectorStore()
    return _singleton


def _stable_uuid(seed: str) -> str:
    """稳定 UUID：把 seed 哈希成 UUID 字符串，保证同 memoryId 反复 upsert 得到同一 UUID。"""
    import hashlib

    h = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"