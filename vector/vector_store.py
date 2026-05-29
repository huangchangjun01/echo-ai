from typing import List, Optional, Dict, Any
import json
import logging
import uuid
from urllib.parse import urlparse

from config.config import WEAVIATE_URL, WEAVIATE_CLASS

try:
    import weaviate
    from weaviate.collections.classes.config import Property, DataType
except Exception:
    weaviate = None
    Property = None
    DataType = None

logger = logging.getLogger(__name__)


class WeaviateVectorStore:
    """Weaviate-backed vector store with client-side embeddings.

    Each stored object has: text, metadata (json), fileId, fileName.
    The Weaviate class name is configurable via WEAVIATE_CLASS env var.
    """

    def __init__(self):
        if weaviate is None:
            raise RuntimeError(
                "weaviate-client v4 is not installed. Please install 'weaviate-client>=4.0.0' to use WeaviateVectorStore")

        self.class_name = WEAVIATE_CLASS
        logger.info(f"[__init__] Initializing WeaviateVectorStore for class: {self.class_name}")

        parsed = urlparse(WEAVIATE_URL)
        self._client = weaviate.connect_to_custom(
            http_host=parsed.hostname or "localhost",
            http_port=parsed.port or 8080,
            http_secure=parsed.scheme == "https",
            grpc_host=parsed.hostname or "localhost",
            grpc_port=50051,
            grpc_secure=False,
            skip_init_checks=True,
        )
        logger.info(f"[__init__] Connected to Weaviate at {WEAVIATE_URL}")
        self._ensure_collection()

    def _ensure_collection(self):
        if not self._client.collections.exists(self.class_name):
            self._client.collections.create(
                name=self.class_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.TEXT),
                    Property(name="fileId", data_type=DataType.TEXT),
                    Property(name="fileName", data_type=DataType.TEXT),
                ],
                vectorizer_config=None,
            )
            logger.info(f"[_ensure_collection] Created collection: {self.class_name}")
        else:
            logger.info(f"[_ensure_collection] Collection {self.class_name} already exists")

    def _get_collection(self):
        return self._client.collections.get(self.class_name)

    def _insert_one(self, coll, id_: str, text: str, md: Optional[Dict], vector):
        """Insert a single document, returning UUID."""
        file_id = md.get("fileId") if isinstance(md, dict) else None
        file_name = md.get("fileName") if isinstance(md, dict) else None
        try:
            id_uuid = uuid.UUID(id_)
        except Exception:
            id_uuid = uuid.uuid4()
            logger.warning(f"[add_texts] Invalid id '{id_}', generated UUID: {id_uuid}")
        coll.data.insert(
            properties={"text": text, "metadata": json.dumps(md) if md else "", "fileId": file_id,
                        "fileName": file_name},
            uuid=str(id_uuid),
            vector=vector,
        )

    def add_texts(self, ids: List[str], texts: List[str],
                  metadatas: Optional[List[Dict[str, Any]]] = None,
                  embeddings: Optional[List[List[float]]] = None):
        if embeddings is None:
            raise SyntaxError('Embeddings is null')
        metadatas = metadatas or [None] * len(texts)
        coll = self._get_collection()
        for id_, text, md, vector in zip(ids, texts, metadatas, embeddings):
            if vector is None:
                logger.error(f"[add_texts] Missing embedding for id={id_}")
                raise ValueError("WeaviateVectorStore requires embeddings.")
            try:
                self._insert_one(coll, id_, text, md, vector)
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

        coll = self._get_collection()
        result = coll.query.near_vector(
            near_vector=q_emb,
            limit=n_results,
            return_properties=["text", "metadata", "fileId", "fileName"],
        )

        ids, docs, mds = [], [], []
        for obj in result.objects:
            ids.append(str(obj.uuid))
            props = obj.properties
            docs.append(props.get("text"))
            try:
                meta = json.loads(props.get("metadata") or "{}")
            except Exception:
                meta = {}
            for key in ("fileId", "fileName"):
                val = props.get(key)
                if val:
                    meta.setdefault(key, val)
            mds.append(meta)

        logger.info(f"[query] Returned {len(ids)} results")
        return {"ids": [ids], "documents": [docs], "metadatas": [mds]}

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            coll = self._get_collection()
            obj = coll.data.get_by_id(uuid=doc_id)
            return obj.properties if obj else None
        except Exception as e:
            logger.error(f"[get_document] Failed for id={doc_id}: {e}")
            return None

    def delete(self, ids: List[str]):
        coll = self._get_collection()
        for id_ in ids:
            try:
                coll.data.delete_by_id(uuid=id_)
                logger.info(f"[delete] Deleted id={id_}")
            except Exception as e:
                logger.error(f"[delete] Failed to delete id={id_}: {e}")


def get_vector_store():
    """Factory returning a WeaviateVectorStore."""
    if weaviate is None:
        raise RuntimeError('weaviate-client not installed.')
    try:
        return WeaviateVectorStore()
    except Exception as e:
        logger.error(f"[get_vector_store] Failed: {e}")
        raise RuntimeError(f'Failed to get Weaviate: {e}')
