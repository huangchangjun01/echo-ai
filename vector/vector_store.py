from __future__ import annotations

import json
import logging
import threading
import time
import uuid as _uuid
from collections.abc import Sequence
from typing import Any

import httpx
from cachetools import TTLCache

from config.config import get_settings
from utils.request_context import log_exception, log_silent_failure, merge_extra

logger = logging.getLogger(__name__)


def _parse_url(url: str) -> tuple[str, int, bool]:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 8080)
    secure = parsed.scheme == "https"
    return host, port, secure


def _ensure_uuid(value: str) -> str:
    try:
        _uuid.UUID(value)
        return value
    except (ValueError, AttributeError, TypeError):
        return str(_uuid.uuid4())


def _where_clause_gql(where: dict[str, Any]) -> str:
    """Render a Chroma-style {field: value} dict as a Weaviate GraphQL `where:` literal.

    Includes a leading comma so the caller doesn't have to manage argument
    punctuation. Returns "" when there are no operands.
    """
    if not where:
        return ""
    ordered_keys = ["userId"] + [k for k in where if k != "userId"]
    operands: list[str] = []
    for field in ordered_keys:
        if field not in where:
            continue
        value = where[field]
        if value is None or value == "":
            continue
        operands.append(
            f"{{ path: [\"{field}\"], operator: Equal, valueText: \"{_escape(value)}\" }}"
        )
    if not operands:
        return ""
    if len(operands) == 1:
        return f", where: {operands[0]}"
    out = operands[0]
    for op in operands[1:]:
        out = f"{{ operator: And, operands: [{out}, {op}] }}"
    return f", where: {out}"


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


# Default document fields for the document store (EchoDoc).
# Memory store uses a different schema and passes its own list.
_DEFAULT_DOC_FIELDS = (
    "text metadata fileId fileName userId chunkIndex _additional { id distance }"
)


def _build_search_query(
    class_name: str,
    vector: list[float],
    limit: int,
    where: dict[str, Any] | None,
    fields: str | tuple[str, ...] | None = None,
) -> str:
    """Build a nearVector Get query.

    `fields` lets callers override the property list (e.g. the memory store
    uses a different schema without fileId/fileName/chunkIndex). When None,
    falls back to the document-store default.
    """
    where_block = _where_clause_gql(where) if where else ""
    vector_lit = _format_vector(vector)
    if fields is None:
        field_block = _DEFAULT_DOC_FIELDS
    elif isinstance(fields, tuple):
        field_block = " ".join(fields)
    else:
        field_block = str(fields)
    return (
        f"{{ Get {{ {class_name}("
        f"nearVector: {{vector: {vector_lit}}}, "
        f"limit: {int(limit)}"
        f"{where_block}"
        f") {{ {field_block} }} }} }}"
    )


def _format_vector(vector: Sequence[float]) -> str:
    return "[" + ",".join(_format_number(x) for x in vector) + "]"


def _format_number(x: Any) -> str:
    if isinstance(x, int):
        return str(x)
    f = float(x)
    if f != f:  # NaN
        return "0"
    if f == float("inf"):
        return "1e308"
    if f == float("-inf"):
        return "-1e308"
    return repr(f)


def _extract_get_payload(response_json: dict[str, Any], class_name: str) -> list[dict[str, Any]]:
    """Pull out the Get[class_name] list from a Weaviate GraphQL response, normalizing errors."""
    errors = response_json.get("errors")
    if errors:
        raise RuntimeError(f"Weaviate GraphQL errors: {errors}")
    data = response_json.get("data") or {}
    get_block = data.get("Get") or {}
    items = get_block.get(class_name) or []
    if not isinstance(items, list):
        raise RuntimeError(f"Unexpected Weaviate response shape: {items!r}")
    return items


class _WeaviateHttpClient:
    """Thin REST wrapper. The weaviate v4 client hard-codes gRPC for queries, which
    breaks against any deployment that does not expose the gRPC port (50051).
    Going through REST/GraphQL keeps the same behaviour with one fewer moving part.
    """

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 30.0):
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        self._owns_client = True

    @classmethod
    def from_settings(cls) -> "_WeaviateHttpClient":
        settings = get_settings().weaviate
        host, port, secure = _parse_url(settings.resolved_url())
        scheme = "https" if secure else "http"
        return cls(f"{scheme}://{host}:{port}", api_key=settings.api_key)

    def close(self) -> None:
        if self._owns_client:
            try:
                self._client.close()
            except Exception as e:
                log_silent_failure(
                    logger,
                    "close Weaviate client failed",
                    exc=e,
                    stage="vector_close",
                    event="client_close_error",
                    class_name=self.class_name,
                )

    # ---------- schema ----------

    def class_exists(self, class_name: str) -> bool:
        r = self._client.get(f"/v1/schema/{class_name}")
        if r.status_code == 200:
            return True
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return False

    def get_class(self, class_name: str) -> dict[str, Any] | None:
        r = self._client.get(f"/v1/schema/{class_name}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def create_class(self, class_obj: dict[str, Any]) -> None:
        r = self._client.post("/v1/schema", json=class_obj)
        if r.status_code >= 400:
            raise RuntimeError(f"create_class failed: {r.status_code} {r.text}")

    def add_property(self, class_name: str, prop: dict[str, Any]) -> bool:
        r = self._client.post(f"/v1/schema/{class_name}/properties", json=prop)
        if 200 <= r.status_code < 300:
            return True
        if r.status_code == 422:
            # property already exists
            return False
        raise RuntimeError(f"add_property failed: {r.status_code} {r.text}")

    # ---------- objects ----------

    def batch_insert(self, objects: list[dict[str, Any]]) -> list[str]:
        if not objects:
            return []
        payload = {"fields": ["ALL"], "objects": objects}
        r = self._client.post("/v1/batch/objects", json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"batch_insert failed: {r.status_code} {r.text}")
        body = r.json()
        uuids: list[str] = []
        for i, item in enumerate(body or []):
            if isinstance(item, dict):
                result = item.get("result", {}) or {}
                status = (result.get("status") or "").lower()
                if status and status != "success":
                    errors = result.get("errors") or {}
                    logger.warning(
                        "batch insert item %s status=%s errors=%s",
                        i, status, errors,
                    )
                uuids.append(str(item.get("id") or objects[i].get("id") or ""))
            else:
                uuids.append(str(objects[i].get("id") or ""))
        return uuids

    def get_object(self, class_name: str, doc_id: str) -> dict[str, Any] | None:
        r = self._client.get(f"/v1/objects/{class_name}/{doc_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        body = r.json()
        return body.get("properties") or {}

    def delete_object(self, class_name: str, doc_id: str) -> None:
        r = self._client.delete(f"/v1/objects/{class_name}/{doc_id}")
        if r.status_code not in (200, 204, 404):
            r.raise_for_status()

    # ---------- search ----------

    def graphql(self, query: str) -> dict[str, Any]:
        r = self._client.post("/v1/graphql", json={"query": query})
        if r.status_code >= 400:
            raise RuntimeError(f"graphql failed: {r.status_code} {r.text}")
        return r.json()


class WeaviateVectorStore:
    """Weaviate vector store backed by the REST API (gRPC-free).

    Each stored object has: text, metadata (json), fileId, fileName, userId, chunkIndex.
    """

    def __init__(self, client: Any | None = None, enable_cache: bool = True):
        settings = get_settings()
        self.class_name: str = settings.weaviate.class_name
        # EchoDoc 存的是 Chinese-CLIP 512 维向量，跨模态 cosine 通常 0.20~0.45；
        # 共用 0.7 会过滤掉所有命中。这里用 doc_threshold 兜底，老字段 vector_similarity_threshold 仍兼容。
        self.threshold: float = (
            settings.weaviate.doc_threshold
            if getattr(settings.weaviate, "doc_threshold", None) is not None
            else getattr(settings, "doc_similarity_threshold", None)
            or settings.vector_similarity_threshold
        )
        self.enable_cache = enable_cache
        self._cache: TTLCache | None = (
            TTLCache(maxsize=settings.ingest.cache_maxsize, ttl=settings.ingest.cache_ttl_seconds)
            if enable_cache
            else None
        )
        self._cache_lock = threading.Lock()

        if client is not None and hasattr(client, "class_exists"):
            self._http = client
        else:
            self._http = _WeaviateHttpClient.from_settings()

        logger.info("Connected to Weaviate (REST) at %s", get_settings().weaviate.resolved_url())
        self._ensure_collection()

    # ---------- collection management ----------

    def _ensure_collection(self) -> None:
        if self._http.class_exists(self.class_name):
            self._ensure_filterable_properties()
            return

        logger.info("Creating collection %s", self.class_name)
        class_obj = {
            "class": self.class_name,
            "vectorizer": "none",
            "properties": [dict(p) for p in self._EXPECTED_PROPERTIES],
        }
        self._http.create_class(class_obj)

    _EXPECTED_PROPERTIES: tuple[dict[str, Any], ...] = (
        {"name": "text", "dataType": ["text"]},
        {"name": "metadata", "dataType": ["text"]},
        {"name": "fileId", "dataType": ["text"], "indexFilterable": True},
        {"name": "fileName", "dataType": ["text"]},
        {"name": "userId", "dataType": ["text"], "indexFilterable": True},
        {"name": "chunkIndex", "dataType": ["int"]},
    )

    def _ensure_filterable_properties(self) -> None:
        """Add any properties that older collections are missing.

        This is a best-effort schema sync so we can roll out new fields without
        requiring a destructive re-import. Properties are created in order; if
        one fails we keep going.
        """
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
                    "add_property failed (skipped)",
                    exc=e,
                    level=logging.WARNING,
                    stage="vector_schema",
                    event="add_prop_error",
                    class_name=self.class_name,
                    prop_name=prop["name"],
                )

    # ---------- write ----------

    def add_texts(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any] | None] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        if embeddings is None:
            raise ValueError("WeaviateVectorStore requires embeddings to be supplied by caller.")
        if len(ids) != len(texts) or len(ids) != len(embeddings):
            raise ValueError("ids/texts/embeddings must have equal length")
        if metadatas is None:
            metadatas = [None] * len(texts)

        objects: list[dict[str, Any]] = []
        returned_ids: list[str] = []
        for id_, text, md, vector in zip(ids, texts, metadatas, embeddings):
            if vector is None:
                raise ValueError(f"Missing embedding for id={id_}")
            md = md or {}
            uuid = _ensure_uuid(id_)
            obj = {
                "id": uuid,
                "class": self.class_name,
                "properties": {
                    "text": text,
                    "metadata": json.dumps(md, ensure_ascii=False),
                    "fileId": md.get("fileId", ""),
                    "fileName": md.get("fileName", ""),
                    "userId": md.get("userId", ""),
                    "chunkIndex": int(md.get("chunkIndex", 0) or 0),
                },
                "vector": list(map(float, vector)),
            }
            objects.append(obj)
            returned_ids.append(uuid)

        t0 = time.perf_counter()
        returned = self._http.batch_insert(objects)
        insert_ms = round((time.perf_counter() - t0) * 1000, 2)
        if self._cache is not None:
            with self._cache_lock:
                self._cache.clear()
        inserted_count = len(returned or returned_ids)
        logger.info(
            "vector add_texts",
            extra=merge_extra(
                stage="vector_add",
                event="ok",
                class_name=self.class_name,
                count=inserted_count,
                http_ms=insert_ms,
            ),
        )
        return returned or returned_ids

    # ---------- read ----------

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        embedding_fn: Any | None = None,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Top-k semantic search scoped to a single tenant.

        `where` MUST contain a non-empty `userId` entry; this is the multi-tenant
        isolation boundary. Pass it explicitly rather than reading from globals so
        that there is no way to accidentally return another tenant's hits.

        Returns a dict shaped like Chroma's response so existing callers don't need to adapt:
            {"ids": [[...]], "documents": [[...]], "metadatas": [[...]], "distances": [[...]]}
        """
        t_total = time.perf_counter()
        if embedding_fn is None:
            raise ValueError("query() requires an embedding_fn")
        if not query_text:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        tenant_id = (where or {}).get("userId")
        if not tenant_id:
            raise ValueError(
                "query() requires `where['userId']` for tenant isolation. "
                "Refusing to run an unscoped query."
            )

        cache_key = None
        cache_hit = False
        if self._cache is not None:
            cache_key = (query_text, int(n_results), json.dumps(where or {}, sort_keys=True))
            cached = self._cache.get(cache_key)
            if cached is not None:
                cache_hit = True
                logger.info(
                    "vector query cache hit",
                    extra=merge_extra(
                        stage="vector_query",
                        event="cache_hit",
                        class_name=self.class_name,
                        n_results=n_results,
                        tenant=tenant_id,
                        query_preview=(query_text or "")[:80],
                    ),
                )
                return cached

        try:
            t0 = time.perf_counter()
            q_emb = embedding_fn([query_text])[0]
            embed_ms = round((time.perf_counter() - t0) * 1000, 2)
        except Exception as e:
            log_exception(
                logger,
                "embedding_fn call failed",
                exc=e,
                stage="vector_query",
                event="embed_error",
                class_name=self.class_name,
                tenant=tenant_id,
                query_preview=(query_text or "")[:80],
            )
            raise ValueError("embedding_fn call failed.") from e

        fetch_limit = max(int(n_results) * 10, 20)
        query = _build_search_query(self.class_name, list(q_emb), fetch_limit, where)

        try:
            t1 = time.perf_counter()
            payload = self._http.graphql(query)
            http_ms = round((time.perf_counter() - t1) * 1000, 2)
            items = _extract_get_payload(payload, self.class_name)
        except Exception as e:
            log_exception(
                logger,
                "Weaviate query failed",
                exc=e,
                stage="vector_query",
                event="http_error",
                class_name=self.class_name,
                tenant=tenant_id,
                query_preview=(query_text or "")[:80],
                http_ms=http_ms if 'http_ms' in locals() else None,
            )
            raise

        scored: list[tuple[float, str, dict[str, Any]]] = []
        for item in items:
            additional = item.get("_additional") or {}
            distance = additional.get("distance")
            if distance is None:
                continue
            similarity = 1.0 - float(distance)
            if similarity >= self.threshold:
                obj_id = additional.get("id") or item.get("id") or ""
                scored.append((similarity, str(obj_id), item))

        scored.sort(key=lambda x: x[0], reverse=True)
        raw_hits = len(scored)
        scored = scored[: int(n_results)]

        ids: list[str] = []
        docs: list[str] = []
        mds: list[dict[str, Any]] = []
        distances: list[float] = []
        score_dist: list[float] = []
        for similarity, obj_id, item in scored:
            ids.append(obj_id)
            docs.append(item.get("text", ""))
            try:
                meta = json.loads(item.get("metadata") or "{}")
            except Exception:
                meta = {}
            for key in ("fileId", "fileName", "userId", "chunkIndex"):
                if key in item and key not in meta:
                    meta[key] = item[key]
            meta["similarity"] = round(similarity, 4)
            mds.append(meta)
            distances.append(round(1.0 - similarity, 6))
            score_dist.append(round(similarity, 4))

        result = {"ids": [ids], "documents": [docs], "metadatas": [mds], "distances": [distances]}
        if cache_key is not None and self._cache is not None:
            with self._cache_lock:
                self._cache[cache_key] = result
        top_score = max(score_dist) if score_dist else 0.0
        median_score = sorted(score_dist)[len(score_dist) // 2] if score_dist else 0.0
        logger.info(
            "vector query",
            extra=merge_extra(
                stage="vector_query",
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
                cache_hit=cache_hit,
                embed_ms=embed_ms,
                http_ms=http_ms,
                duration_ms=round((time.perf_counter() - t_total) * 1000, 2),
                query_preview=(query_text or "")[:80],
            ),
        )
        return result

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        try:
            return self._http.get_object(self.class_name, doc_id)
        except Exception as e:
            log_exception(
                logger,
                "get_document failed",
                exc=e,
                stage="vector_query",
                event="get_doc_error",
                class_name=self.class_name,
                doc_id=doc_id,
            )
            return None

    def delete(self, ids: Sequence[str]) -> None:
        for id_ in ids:
            try:
                self._http.delete_object(self.class_name, id_)
                logger.info("Deleted id=%s", id_)
            except Exception as e:
                log_exception(
                    logger,
                    "delete_object failed",
                    exc=e,
                    stage="vector_write",
                    event="delete_error",
                    class_name=self.class_name,
                    doc_id=id_,
                )
        if self._cache is not None:
            with self._cache_lock:
                self._cache.clear()

    def close(self) -> None:
        try:
            self._http.close()
        except Exception as e:
            log_silent_failure(
                logger,
                "close Weaviate client failed",
                exc=e,
                stage="vector_close",
                event="client_close_error",
                class_name=self.class_name,
            )
        logger.info("Weaviate client closed")


_singleton_lock = threading.Lock()
_singleton: WeaviateVectorStore | None = None


def get_vector_store() -> WeaviateVectorStore:
    """Process-wide singleton; safe under concurrent first-access."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = WeaviateVectorStore()
    return _singleton


def reset_vector_store() -> None:
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            try:
                _singleton.close()
            except Exception as e:
                log_silent_failure(
                    logger,
                    "reset_vector_store: close failed",
                    exc=e,
                    stage="vector_close",
                    event="reset_error",
                )
        _singleton = None
