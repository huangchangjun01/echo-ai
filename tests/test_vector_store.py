"""Unit tests for WeaviateVectorStore using a fake REST client."""
from __future__ import annotations

import re
from typing import Any


def _parse_where_from_query(query: str) -> dict[str, Any]:
    """Extract field→value pairs from the GraphQL `where:` literal produced by _where_clause_gql."""
    out: dict[str, Any] = {}
    for m in re.finditer(r'path:\s*\["(\w+)"\].*?valueText:\s*"([^"]+)"', query):
        out[m.group(1)] = m.group(2)
    return out


def _extract_class_name(query: str) -> str:
    m = re.search(r"Get\s*{\s*(\w+)\s*\(", query)
    return m.group(1) if m else ""


def _matches(props: dict[str, Any], where: dict[str, Any]) -> bool:
    return all(str(props.get(k, "")) == str(v) for k, v in where.items())


class _FakeHttpClient:
    """Stand-in for _WeaviateHttpClient: stores objects in-memory and serves fake GraphQL."""

    def __init__(self):
        self.objects: list[dict[str, Any]] = []
        self.classes: dict[str, dict[str, Any]] = {}
        self.graphql_calls: list[str] = []
        self.batch_calls: list[list[dict[str, Any]]] = []
        self._next_uuid = 0

    def class_exists(self, name: str) -> bool:
        return name in self.classes

    def get_class(self, name: str) -> dict[str, Any] | None:
        return self.classes.get(name)

    def create_class(self, obj: dict[str, Any]) -> None:
        self.classes[obj["class"]] = obj

    def add_property(self, name: str, prop: dict[str, Any]) -> bool:
        cls = self.classes.get(name)
        if not cls:
            return False
        existing = {p.get("name") for p in cls.get("properties", []) or []}
        if prop["name"] in existing:
            return False
        cls.setdefault("properties", []).append(prop)
        return True

    def batch_insert(self, objects: list[dict[str, Any]]) -> list[str]:
        self.batch_calls.append(objects)
        ids: list[str] = []
        for obj in objects:
            uid = obj.get("id") or f"uuid-{self._next_uuid}"
            self._next_uuid += 1
            self.objects.append({"id": uid, **obj})
            ids.append(uid)
        return ids

    def graphql(self, query: str) -> dict[str, Any]:
        self.graphql_calls.append(query)
        where = _parse_where_from_query(query)
        cls_name = _extract_class_name(query)
        results: list[dict[str, Any]] = []
        for obj in self.objects:
            props = obj.get("properties", {}) or {}
            if cls_name and obj.get("class") and obj["class"] != cls_name:
                continue
            if not _matches(props, where):
                continue
            distance = obj.get("_distance", 0.1)
            results.append(
                {
                    "text": props.get("text", ""),
                    "metadata": props.get("metadata", ""),
                    "fileId": props.get("fileId", ""),
                    "fileName": props.get("fileName", ""),
                    "userId": props.get("userId", ""),
                    "chunkIndex": props.get("chunkIndex", 0),
                    "_additional": {"id": obj["id"], "distance": distance},
                }
            )
        return {"data": {"Get": {cls_name: results}}}

    def get_object(self, class_name: str, doc_id: str) -> dict[str, Any] | None:
        for obj in self.objects:
            if obj["id"] == doc_id:
                return obj.get("properties") or {}
        return None

    def delete_object(self, class_name: str, doc_id: str) -> None:
        self.objects = [o for o in self.objects if o["id"] != doc_id]

    def close(self) -> None:
        pass


def _seed_objects() -> dict[str, list[dict[str, Any]]]:
    """The multi-tenant dataset used by the isolation tests.

    Distances are chosen so that, with the test store's `threshold = 0.5`,
    both u1 objects pass while still keeping u1-u1 ordered by similarity
    (0.1 → 0.9 similarity beats 0.3 → 0.7 similarity).
    """
    return {
        "u1": [
            {
                "id": "real-uuid-1",
                "class": "EchoDoc",
                "properties": {
                    "text": "hello world",
                    "metadata": '{"fileId": "f1", "userId": "u1"}',
                    "fileId": "f1",
                    "fileName": "f1.txt",
                    "userId": "u1",
                    "chunkIndex": 0,
                },
                "vector": [0.1, 0.2],
                "_distance": 0.1,
            },
            {
                "id": "real-uuid-2",
                "class": "EchoDoc",
                "properties": {
                    "text": "another doc",
                    "metadata": '{"fileId": "f2", "userId": "u1"}',
                    "fileId": "f2",
                    "fileName": "f2.txt",
                    "userId": "u1",
                    "chunkIndex": 0,
                },
                "vector": [0.3, 0.4],
                "_distance": 0.3,
            },
        ],
        "u2": [
            {
                "id": "real-uuid-3",
                "class": "EchoDoc",
                "properties": {
                    "text": "u2 secret",
                    "metadata": '{"fileId": "f3", "userId": "u2"}',
                    "fileId": "f3",
                    "fileName": "f3.txt",
                    "userId": "u2",
                    "chunkIndex": 0,
                },
                "vector": [0.5, 0.6],
                "_distance": 0.05,
            },
        ],
    }


def _make_store(enable_cache: bool = False) -> Any:
    from vector import vector_store as vs_mod

    # Replace _ensure_uuid so add_texts() produces predictable "uuid-N" identifiers
    # (the real path calls uuid4() for invalid input; tests want a stable prefix).
    counter = {"n": 0}

    def _fake_ensure_uuid(value: str) -> str:
        counter["n"] += 1
        return f"uuid-{counter['n'] - 1}"

    vs_mod._ensure_uuid = _fake_ensure_uuid  # type: ignore[assignment]

    fake_http = _FakeHttpClient()
    fake_http.classes["EchoDoc"] = {
        "class": "EchoDoc",
        "vectorizer": "none",
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "metadata", "dataType": ["text"]},
            {"name": "fileId", "dataType": ["text"], "indexFilterable": True},
            {"name": "fileName", "dataType": ["text"]},
            {"name": "userId", "dataType": ["text"], "indexFilterable": True},
            {"name": "chunkIndex", "dataType": ["int"]},
        ],
    }

    for user_objs in _seed_objects().values():
        for obj in user_objs:
            fake_http.objects.append(obj)

    store = vs_mod.WeaviateVectorStore(client=fake_http, enable_cache=enable_cache)
    # Lower the threshold so both seeded u1 objects pass (similarity 0.9 and 0.7).
    store.threshold = 0.5
    return store


def test_query_requires_user_id_for_tenant_isolation():
    store = _make_store()
    with __import__("pytest").raises(ValueError, match="userId"):
        store.query(
            "hi",
            n_results=2,
            embedding_fn=lambda texts: [[0.1, 0.2]],
            where=None,
        )


def test_query_filters_by_user_id():
    store = _make_store()
    result = store.query(
        "hi",
        n_results=5,
        embedding_fn=lambda texts: [[0.1, 0.2]],
        where={"userId": "u1"},
    )
    ids = result["ids"][0]
    docs = result["documents"][0]
    assert ids == ["real-uuid-1", "real-uuid-2"]
    assert docs == ["hello world", "another doc"]
    # Sanity: u2's data is not present even though it has a closer distance.
    assert "u2 secret" not in docs


def test_query_isolates_tenants():
    store = _make_store()
    a = store.query("hi", n_results=5, embedding_fn=lambda texts: [[0.1]], where={"userId": "u1"})
    b = store.query("hi", n_results=5, embedding_fn=lambda texts: [[0.1]], where={"userId": "u2"})
    a_docs = a["documents"][0]
    b_docs = b["documents"][0]
    assert a_docs == ["hello world", "another doc"]
    assert b_docs == ["u2 secret"]
    assert set(a["ids"][0]).isdisjoint(set(b["ids"][0]))


def test_query_passes_userid_first_for_index_usage():
    store = _make_store()
    fake_http = store._http
    fake_http.graphql_calls.clear()
    store.query("hi", n_results=5, embedding_fn=lambda texts: [[0.1]], where={"userId": "u1"})
    assert fake_http.graphql_calls, "expected a GraphQL call"
    last_query = fake_http.graphql_calls[-1]
    userid_pos = last_query.find('path: ["userId"]')
    assert userid_pos != -1, "userId filter must be present in the GraphQL query"
    fileid_pos = last_query.find('path: ["fileId"]')
    if fileid_pos != -1:
        assert userid_pos < fileid_pos, "userId must appear before other filters so the index is usable"


def test_query_returns_topk_with_real_uuids():
    store = _make_store()
    result = store.query(
        "hi",
        n_results=2,
        embedding_fn=lambda texts: [[0.1, 0.2]],
        where={"userId": "u1"},
    )
    ids = result["ids"][0]
    docs = result["documents"][0]
    mds = result["metadatas"][0]

    assert len(ids) == 2, "should return top-k (k=2), not just top-1"
    assert ids[0] == "real-uuid-1", "real Weaviate UUID must be returned, not a fresh uuid4"
    assert docs[0] == "hello world"
    assert mds[0]["fileId"] == "f1"
    assert "similarity" in mds[0]


def test_query_respects_threshold():
    store = _make_store()
    store.threshold = 0.9

    result = store.query(
        "hi",
        n_results=5,
        embedding_fn=lambda texts: [[0.0]],
        where={"userId": "u1"},
    )
    assert len(result["ids"][0]) == 1
    assert result["ids"][0][0] == "real-uuid-1"


def test_query_cache_keys_include_user_id():
    store = _make_store(enable_cache=True)

    call_count = {"n": 0}

    def embedding_fn(texts):
        call_count["n"] += 1
        return [[0.1, 0.2, 0.3]]

    a1 = store.query("hi", n_results=2, embedding_fn=embedding_fn, where={"userId": "u1"})
    a2 = store.query("hi", n_results=2, embedding_fn=embedding_fn, where={"userId": "u1"})
    b1 = store.query("hi", n_results=2, embedding_fn=embedding_fn, where={"userId": "u2"})
    assert a1["ids"] == a2["ids"]
    assert a1["ids"] != b1["ids"], "same query for different tenants must not share a cache entry"
    # Two distinct cache keys => embedding_fn called twice (once per tenant).
    assert call_count["n"] == 2


def test_add_texts_uses_batch():
    store = _make_store()
    uuids = store.add_texts(
        ids=["a", "b"],
        texts=["x", "y"],
        metadatas=[{"fileId": "a", "userId": "u1"}, {"fileId": "b", "userId": "u1"}],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
    )
    assert all(u.startswith("uuid-") for u in uuids)