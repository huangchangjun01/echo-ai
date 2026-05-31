from typing import List, Optional, Dict, Any
import json
import logging
import uuid
from urllib.parse import urlparse

from config.config import WEAVIATE_URL, WEAVIATE_CLASS, VECTOR_SIMILARITY_THRESHOLD

try:
    import weaviate
except Exception:
    weaviate = None

logger = logging.getLogger(__name__)


def _parse_url(url: str) -> tuple:
    """Parse WEAVIATE_URL into host and port."""
    parsed = urlparse(url)
    host = parsed.hostname or 'localhost'
    port = parsed.port or 8080
    secure = parsed.scheme == 'https'
    return host, port, secure


class WeaviateVectorStore:
    """Weaviate-backed vector store with client-side embeddings.

    Each stored object has: text, metadata (json), fileId, fileName.
    The Weaviate class name is configurable via WEAVIATE_CLASS env var.
    """

    def __init__(self):
        if weaviate is None:
            raise RuntimeError(
                "weaviate-client is not installed. Please install 'weaviate-client>=3.26.7' to use WeaviateVectorStore")

        self.class_name = WEAVIATE_CLASS
        logger.info(f"[__init__] Initializing WeaviateVectorStore for class: {self.class_name}")

        host, port, secure = _parse_url(WEAVIATE_URL)
        url = f"{'https' if secure else 'http'}://{host}:{port}"
        self._client = weaviate.Client(url=url)
        logger.info(f"[__init__] Connected to Weaviate at {url}")
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            schema = self._client.schema.get()
            classes = [c.get('class') for c in schema.get('classes', [])]
            if self.class_name not in classes:
                raise ValueError(f"Collection {self.class_name} does not exist")
            logger.info(f"[_ensure_collection] Collection {self.class_name} already exists")
        except Exception:
            self._client.schema.create_class(
                class_name=self.class_name,
                properties=[
                    {"name": "text", "dataType": ["text"]},
                    {"name": "metadata", "dataType": ["text"]},
                    {"name": "fileId", "dataType": ["text"]},
                    {"name": "fileName", "dataType": ["text"]},
                ],
                vectorizer_config=None,
            )
            logger.info(f"[_ensure_collection] Created collection: {self.class_name}")

    def _insert_one(self, id_: str, text: str, md: Optional[Dict], vector):
        """Insert a single document, returning UUID."""
        file_id = md.get("fileId") if isinstance(md, dict) else None
        file_name = md.get("fileName") if isinstance(md, dict) else None
        try:
            id_uuid = uuid.UUID(id_)
        except Exception:
            id_uuid = uuid.uuid4()
            logger.warning(f"[add_texts] Invalid id '{id_}', generated UUID: {id_uuid}")
        self._client.data_object.create(
            class_name=self.class_name,
            data_object={
                "text": text,
                "metadata": json.dumps(md) if md else "",
                "fileId": file_id,
                "fileName": file_name,
            },
            uuid=str(id_uuid),
            vector=vector,
        )

    def add_texts(self, ids: List[str], texts: List[str],
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  embeddings: Optional[List[List[float]]] = None):
        if embeddings is None:
            raise SyntaxError('Embeddings is null')
        metadatas = metadatas or [None] * len(texts)
        for id_, text, md, vector in zip(ids, texts, metadatas, embeddings):
            if vector is None:
                logger.error(f"[add_texts] Missing embedding for id={id_}")
                raise ValueError("WeaviateVectorStore requires embeddings.")
            try:
                self._insert_one(id_, text, md, vector)
                logger.info(f"[add_texts] Inserted id={id_}")
            except Exception as e:
                logger.error(f"[add_texts] Failed to insert id={id_}: {e}")
                raise

    def query(self, query_text: str, n_results: int = 5, embedding_fn: Optional[Any] = None) -> Dict[str, Any]:
        if embedding_fn is None:
            raise ValueError("query() requires an embedding_fn to compute query embedding.")
        try:
            q_emb = embedding_fn([query_text])[0]
        except Exception as e:
            logger.error(f"[query] Failed to compute query embedding: {e}")
            raise ValueError("embedding_fn call failed.") from e

        # Fetch more results to account for filtering by similarity threshold
        fetch_limit = max(n_results * 10, 20)
        result = self._client.query.get(
            self.class_name,
            ['text', 'metadata', 'fileId', 'fileName']
        ).with_near_vector({'vector': q_emb}).with_additional(['distance']).with_limit(fetch_limit).do()

        items = result.get('data', {}).get('Get', {}).get(self.class_name, [])

        # Filter by similarity threshold using distance (1 - distance = cosine similarity)
        scored_items = []
        for item in items:
            distance = item.get('_additional', {}).get('distance')
            if distance is None:
                logger.warning(f"[query] No distance returned for item, skipping")
                continue

            # Cosine similarity = 1 - cosine distance
            similarity = 1.0 - distance
            if similarity >= VECTOR_SIMILARITY_THRESHOLD:
                scored_items.append((similarity, item))

        # Sort by similarity descending
        scored_items.sort(key=lambda x: x[0], reverse=True)

        # Return only the top result (most similar)
        ids, docs, mds = [], [], []
        if scored_items:
            top_similarity, top_item = scored_items[0]
            ids.append(str(uuid.uuid4()))
            docs.append(top_item.get('text'))
            try:
                meta = json.loads(top_item.get('metadata') or "{}")
            except Exception:
                meta = {}
            for key in ("fileId", "fileName"):
                val = top_item.get(key)
                if val:
                    meta.setdefault(key, val)
            meta['similarity'] = round(top_similarity, 4)
            mds.append(meta)
            logger.info(f"[query] Top result similarity: {top_similarity:.4f}")

        logger.info(f"[query] Returned {len(ids)} results (from {len(items)} fetched, {len(scored_items)} above threshold)")
        return {"ids": [ids], "documents": [docs], "metadatas": [mds]}

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            obj = self._client.data_object.get_by_id(uuid=doc_id, class_name=self.class_name)
            return obj.get('properties') if obj else None
        except Exception as e:
            logger.error(f"[get_document] Failed for id={doc_id}: {e}")
            return None

    def delete(self, ids: List[str]):
        for id_ in ids:
            try:
                self._client.data_object.delete(uuid=id_, class_name=self.class_name)
                logger.info(f"[delete] Deleted id={id_}")
            except Exception as e:
                logger.error(f"[delete] Failed to delete id={id_}: {e}")

    def close(self):
        # v3 client doesn't need explicit close
        logger.info("[close] Weaviate client (v3, no explicit close needed)")


def get_vector_store():
    """Factory returning a WeaviateVectorStore."""
    if weaviate is None:
        raise RuntimeError('weaviate-client not installed.')
    try:
        return WeaviateVectorStore()
    except Exception as e:
        logger.error(f"[get_vector_store] Failed: {e}")
        raise RuntimeError(f'Failed to get Weaviate: {e}')