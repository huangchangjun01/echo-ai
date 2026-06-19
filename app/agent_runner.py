from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from biz.ingest import ingest_file
from config.config import get_settings
from embedding.embeddings import ChineseCLIPEmbeddings
from vector import vector_store as vs_module

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Echo-AI (threshold=%.3f)", settings.vector_similarity_threshold)

    embeddings = ChineseCLIPEmbeddings()
    if settings.embedding.warmup_on_start:
        await asyncio.to_thread(embeddings.warmup)

    try:
        # Look up via the module each time so monkeypatch / test overrides work.
        vectorstore = await asyncio.to_thread(vs_module.get_vector_store)
    except Exception as e:
        logger.error("Vector store init failed: %s", e)
        raise

    app.state.embeddings = embeddings
    app.state.vectorstore = vectorstore
    app.state.settings = settings
    logger.info("Echo-AI ready")
    try:
        yield
    finally:
        try:
            await asyncio.to_thread(vs_module.reset_vector_store)
        except Exception:
            pass
        logger.info("Echo-AI shutdown complete")


app = FastAPI(
    title="Echo-AI Agent",
    version="2.0.0",
    lifespan=lifespan,
)


class EmbeddingsRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=256)


class AddRequest(BaseModel):
    ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, Any]] | None = None


class FileObject(BaseModel):
    fileId: str
    fileName: str
    fileKey: str | None = None
    url: str | None = None


class IngestRequest(BaseModel):
    userId: str
    file: FileObject


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)
    k: int = Field(5, ge=1, le=50)
    user_id: str = Field(..., min_length=1, max_length=128, alias="userId")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)
    user_id: str = Field(..., min_length=1, max_length=128, alias="userId")


def _embedding_fn_for(embeddings: ChineseCLIPEmbeddings):
    def _fn(texts: list[str]) -> list[list[float]]:
        return embeddings.embed_documents(texts)

    return _fn


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    embeddings = request.app.state.embeddings
    vectorstore = request.app.state.vectorstore
    where = {"userId": req.user_id}
    try:
        result = await asyncio.to_thread(
            vectorstore.query,
            req.query,
            req.k,
            _embedding_fn_for(embeddings),
            where,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(e))

    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    mds = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    candidates = [
        {"id": id_, "document": doc, "metadata": md, "distance": d}
        for id_, doc, md, d in zip(ids, docs, mds, distances)
    ]
    return {"query": req.query, "k": req.k, "userId": req.user_id, "candidates": candidates}


@app.post("/search")
async def search(req: SearchRequest, request: Request):
    return await chat(req=ChatRequest(query=req.query, k=req.k, userId=req.user_id), request=request)


@app.post("/ingest_file")
async def ingest_file_endpoint(req: IngestRequest, background: BackgroundTasks, request: Request):
    embeddings = request.app.state.embeddings
    vectorstore = request.app.state.vectorstore
    background.add_task(ingest_file, req.userId, req.file.model_dump(), embeddings, vectorstore)
    return {"ok": True, "queued": True, "fileId": req.file.fileId}


@app.get("/health")
async def health_check():
    """Lightweight health endpoint so the service can be probed by uptime checks."""
    return {"status": "ok"}