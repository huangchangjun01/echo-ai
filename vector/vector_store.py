from typing import List, Optional, Dict, Any
import json
import logging
import uuid
from urllib.parse import urlparse

# Use the project's config module
from config.config import WEAVIATE_URL, WEAVIATE_CLASS

try:
    import weaviate
    from weaviate import WeaviateClient
    from weaviate.collections.classes.config import Property, DataType
    from weaviate.collections.classes.data import DataObject
except Exception:
    weaviate = None
    WeaviateClient = None
    Property = None
    DataType = None

logger = logging.getLogger(__name__)


class WeaviateVectorStore:
    """A minimal Weaviate-backed vector store compatible with the existing method names.

    Methods implemented: add_texts, query, get_document, delete, persist

    This implementation expects embeddings to be supplied by the caller (client-side embeddings).
    Each object stored will have properties: 'text' (string) and 'metadata' (json string).
    The Weaviate class name is configurable via environment variable WEAVIATE_CLASS.
    """

    def __init__(self):
        if weaviate is None:
            raise RuntimeError(
                "weaviate-client v4 is not installed. Please install 'weaviate-client>=4.0.0' to use WeaviateVectorStore")

        self.class_name = WEAVIATE_CLASS
        logger.info(f"[__init__] Initializing WeaviateVectorStore for class: {self.class_name}")

        # Build client using v4 API
        parsed = urlparse(WEAVIATE_URL)
        http_secure = parsed.scheme == "https"
        auth = None

        try:
            self._client = weaviate.connect_to_custom(
                http_host=parsed.hostname or "localhost",
                http_port=parsed.port or 8080,
                http_secure=http_secure,
                grpc_host=parsed.hostname or "localhost",
                grpc_port=50051,
                grpc_secure=False,
                auth_credentials=auth,
                skip_init_checks=True,
            )
            logger.info(f"[__init__] Connected to Weaviate at {WEAVIATE_URL}")
        except Exception as e:
            logger.error(f"[__init__] Failed to connect to Weaviate at {WEAVIATE_URL}: {e}")
            raise

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if not self._client.collections.exists(self.class_name):
            try:
                self._client.collections.create(
                    name=self.class_name,
                    properties=[
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="metadata", data_type=DataType.TEXT),
                        Property(name="fileId", data_type=DataType.TEXT),
                        Property(name="fileName", data_type=DataType.TEXT),
                    ],
                    vectorizer_config=None,  # no automatic vectorization
                )
                logger.info(f"[_ensure_collection] Created collection: {self.class_name}")
            except Exception as e:
                logger.error(f"[_ensure_collection] Failed to create collection {self.class_name}: {e}")
                raise
        else:
            logger.info(f"[_ensure_collection] Collection {self.class_name} already exists")

    def _get_collection(self):
        """Get the collection object."""
        return self._client.collections.get(self.class_name)

    def persist(self):
        # Weaviate persists automatically
        return

    def add_texts(self, ids: List[str], texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None,
                  embeddings: Optional[List[List[float]]] = None):
        if embeddings is None:
            raise SyntaxError('Embeddings is null')

        if metadatas is None:
            metadatas = [None] * len(texts)

        coll = self._get_collection()
        for idx, (id_, text, md) in enumerate(zip(ids, texts, metadatas)):
            file_id = None
            file_name = None
            vector = embeddings[idx]
            if isinstance(md, dict):
                file_id = md.get("fileId")
                file_name = md.get("fileName")
            if vector is None:
                logger.error(f"[add_texts] Missing embedding for id={id_}, text length={len(text)}")
                raise ValueError(
                    "WeaviateVectorStore requires embeddings when vectorizer='none'. Provide embeddings or an embedding_function.")

            # Convert id to UUID format (Weaviate requires UUID type)
            try:
                id_uuid = uuid.UUID(id_)
            except Exception:
                id_uuid = uuid.uuid4()
                logger.warning(f"[add_texts] Invalid id '{id_}' provided, generated new UUID: {id_uuid}")

            try:
                coll.data.insert(
                    properties={
                        "text": text,
                        "metadata": json.dumps(md) if md is not None else "",
                        "fileId": file_id,
                        "fileName": file_name,
                    },
                    uuid=str(id_uuid),
                    vector=vector,
                )
                logger.info(f"[add_texts] Successfully inserted id={id_}, text length={len(text)}")
            except Exception as e:
                logger.error(f"[add_texts] Failed to insert id={id_}, error={e}")
                raise

    def query(self, query_text: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None):
        if self.embedding_function is not None:
            try:
                q_emb = self.embedding_function([query_text])[0]
            except Exception as e:
                logger.error(f"[query] Failed to compute query embedding: {e}")
                raise ValueError(
                    "WeaviateVectorStore requires an embedding function to compute query embeddings when vectorizer='none'.")
        else:
            raise ValueError(
                "WeaviateVectorStore requires an embedding function to compute query embeddings when vectorizer='none'.")

        coll = self._get_collection()
        try:
            result = coll.query.near_vector(
                near_vector=q_emb,
                limit=n_results,
                return_properties=["text", "metadata", "fileId", "fileName"],
            )
        except Exception as e:
            logger.error(f"[query] Failed to query vector store: {e}")
            raise

        ids = []
        docs = []
        mds = []
        for obj in result.objects:
            ids.append(str(obj.uuid))
            props = obj.properties
            docs.append(props.get("text"))
            meta_raw = props.get("metadata")
            try:
                meta = json.loads(meta_raw) if meta_raw else {}
            except Exception:
                meta = meta_raw or {}
            file_id_prop = props.get("fileId")
            file_name_prop = props.get("fileName")
            if isinstance(meta, dict):
                if file_id_prop:
                    meta.setdefault("fileId", file_id_prop)
                if file_name_prop:
                    meta.setdefault("fileName", file_name_prop)
            else:
                meta = {"metadata": meta}
                if file_id_prop:
                    meta["fileId"] = file_id_prop
                if file_name_prop:
                    meta["fileName"] = file_name_prop
            mds.append(meta)

        logger.info(f"[query] Query returned {len(ids)} results")
        return {"ids": [ids], "documents": [docs], "metadatas": [mds]}

    def get_document(self, id: str) -> Optional[Dict[str, Any]]:
        try:
            coll = self._get_collection()
            obj = coll.data.get_by_id(uuid=id)
            if obj:
                logger.info(f"[get_document] Found document id={id}")
                return obj.properties
            logger.warning(f"[get_document] Document not found id={id}")
            return None
        except Exception as e:
            logger.error(f"[get_document] Failed to get document id={id}: {e}")
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
    """Factory that returns a WeaviateVectorStore. This project is Weaviate-only here.

    If `weaviate-client` is not installed, an informative RuntimeError is raised.
    """

    # Try to use the Weaviate client if available and a server is reachable
    if weaviate is not None:
        try:
            return WeaviateVectorStore()
        except Exception as e:
            # If creating the Weaviate store fails (no server, auth issue, etc.), fall back to in-memory store
            logger.error(f"[get_vector_store] Failed to get Weaviate.{e}")
            raise RuntimeError('Failed to get Weaviate. %s' % (e))

    # weaviate client not installed
    raise RuntimeError('Weaviate client not installed. ')
