import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Any, Dict

from embedding.embeddings import ChineseCLIPEmbeddings
from vector.vector_store import get_vector_store
from biz.ingest import ingest_background

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _EMBEDDINGS, _VECTORSTORE
    if _EMBEDDINGS is None:
        _EMBEDDINGS = ChineseCLIPEmbeddings()
    if _VECTORSTORE is None:
        _VECTORSTORE = get_vector_store()
    yield


app = FastAPI(title="Echo-AI LangChain Agent", lifespan=lifespan)

# Globals
_EMBEDDINGS = None
_VECTORSTORE = None


class EmbeddingsRequest(BaseModel):
    texts: List[str]


class AddRequest(BaseModel):
    ids: List[str]
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None


class FileObject(BaseModel):
    fileId: str
    fileName: str
    fileKey: Optional[str] = None
    url: Optional[str] = None


class IngestRequest(BaseModel):
    userId: str
    file: FileObject


class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5


@app.post("/chat")
async def chat(payload: Dict[str, Any]):
    q = payload.get("query")
    k = int(payload.get("k", 5))
    if not q:
        raise HTTPException(status_code=400, detail="missing query")
    res = _VECTORSTORE.query(query_text=q, n_results=k)
    docs = []
    if res:
        docs = [{"id": id_, "document": doc, "metadata": md} for id_, doc, md in
                zip(res.get("ids", [[]])[0], res.get("documents", [[]])[0], res.get("metadatas", [[]])[0])]
    return {"query": q, "candidates": docs}


@app.post('/ingest_file')
async def ingest_file(req: IngestRequest, background: BackgroundTasks):
    """Endpoint to ingest a file referenced by URL or key in Qiniu. Will process in background.

    Request body example:
    {
      "userId": "user-123",
      "file": {"fileId": "f1", "fileName": "image.png", "fileKey": "path/to/image.png"}
    }
    """
    try:
        user_id = req.userId
        # use pydantic v2 model_dump instead of deprecated dict()
        file_obj = req.file.model_dump()
        # enqueue background job; pass current embeddings and vectorstore instances
        background.add_task(ingest_background, user_id, file_obj, _EMBEDDINGS, _VECTORSTORE)
        return {"ok": True, "queued": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Lightweight health endpoint so the service can be probed by uptime checks."""
    return {"status": "ok"}
