from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from config.config import get_settings
from database.models import CoreMemory, MemoryRelation
from database.mysql import get_session
from memory.archiver import get_l0_memories

logger = logging.getLogger(__name__)


async def retrieve_l0_memories(user_id: str) -> list[dict[str, Any]]:
    """加载 L0 核心记忆。"""
    return await get_l0_memories(user_id)


async def retrieve_l1_memories(
    user_id: str,
    query: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    """使用 Weaviate 向量检索 Top-K 相关 L1 记忆。

    优先使用 BGE-M3 文本向量，回退到 Chinese-CLIP。
    """
    from vector.vector_store import get_vector_store, WeaviateVectorStore

    vector_store = get_vector_store()

    # 获取文本向量：优先 BGE-M3，回退 Chinese-CLIP
    try:
        from embedding.bge_m3 import compute_query_embedding
        query_vector = compute_query_embedding(query)
    except Exception:
        logger.warning("BGE-M3 embedding failed, falling back to Chinese-CLIP")
        from embedding.models import compute_text_embedding
        query_vector = compute_text_embedding(query)

    def embedding_fn(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if texts == [query]:
            return [query_vector]
        try:
            from embedding.bge_m3 import compute_embeddings
            return compute_embeddings(texts)
        except Exception:
            from embedding.models import compute_text_embeddings
            return compute_text_embeddings(texts)

    result = vector_store.query(
        query_text=query,
        n_results=k,
        embedding_fn=embedding_fn,
        where={"userId": user_id},
    )

    memories: list[dict[str, Any]] = []
    ids_list = result.get("ids", [[]])
    docs_list = result.get("documents", [[]])
    mds_list = result.get("metadatas", [[]])
    dists_list = result.get("distances", [[]])

    ids = ids_list[0] if ids_list else []
    docs = docs_list[0] if docs_list else []
    mds = mds_list[0] if mds_list else []
    dists = dists_list[0] if dists_list else []

    for i in range(len(ids)):
        memories.append({
            "id": ids[i] if i < len(ids) else "",
            "content": docs[i] if i < len(docs) else "",
            "metadata": mds[i] if i < len(mds) else {},
            "distance": dists[i] if i < len(dists) else 0.0,
        })

    logger.info("retrieve_l1_memories user=%s k=%d returned=%d", user_id, k, len(memories))
    return memories


async def retrieve_causal_chain(
    user_id: str,
    memory_id: int,
) -> list[dict[str, Any]]:
    """从 MySQL memory_relations 表查询因果关系链。

    递归查询以 memory_id 为源的 causes 关系链。
    """
    chain: list[dict[str, Any]] = []
    visited: set[int] = set()

    async def _recurse(source_id: int) -> None:
        if source_id in visited:
            return
        visited.add(source_id)

        async with get_session() as session:
            result = await session.execute(
                select(MemoryRelation, CoreMemory)
                .join(CoreMemory, MemoryRelation.target_id == CoreMemory.id)
                .where(
                    MemoryRelation.source_id == source_id,
                    MemoryRelation.relation_type == "causes",
                )
                .order_by(MemoryRelation.created_at.asc())
            )
            rows = result.all()

        for rel, mem in rows:
            chain.append({
                "relation_id": rel.id,
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "relation_type": rel.relation_type,
                "confidence": rel.confidence,
                "target_content": mem.content,
                "target_layer": mem.layer,
            })
            await _recurse(rel.target_id)

    await _recurse(memory_id)
    logger.info(
        "retrieve_causal_chain user=%s memory_id=%s chain_len=%d",
        user_id, memory_id, len(chain),
    )
    return chain


async def retrieve_cross_modal(
    user_id: str,
    query: str,
    modality: str = "text",
) -> list[dict[str, Any]]:
    """跨模态检索：根据 modality 选择对应的 embedding 模型在 Weaviate 中检索。

    modality: text / image / audio
    """
    from vector.vector_store import get_vector_store

    vector_store = get_vector_store()

    if modality == "image":
        # 图像模态使用 CLIP image embedding
        try:
            from embedding.models import compute_image_embedding
            query_vector = compute_image_embedding(query.encode("utf-8"))
        except Exception as e:
            logger.warning("CLIP image embedding failed: %s, falling back to BGE-M3", e)
            from embedding.bge_m3 import compute_query_embedding
            query_vector = compute_query_embedding(query)
    elif modality == "audio":
        # 音频模态使用 Whisper 特征
        try:
            from embedding.whisper import extract_voiceprint
            query_vector = extract_voiceprint(query)
        except Exception as e:
            logger.warning("Whisper embedding failed: %s, falling back to BGE-M3", e)
            from embedding.bge_m3 import compute_query_embedding
            query_vector = compute_query_embedding(query)
    else:
        # 默认文本模态：BGE-M3
        try:
            from embedding.bge_m3 import compute_query_embedding
            query_vector = compute_query_embedding(query)
        except Exception:
            from embedding.models import compute_text_embedding
            query_vector = compute_text_embedding(query)

    def embedding_fn(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            from embedding.bge_m3 import compute_embeddings
            return compute_embeddings(texts)
        except Exception:
            from embedding.models import compute_text_embeddings
            return compute_text_embeddings(texts)

    result = vector_store.query(
        query_text=query,
        n_results=5,
        embedding_fn=embedding_fn,
        where={"userId": user_id},
    )

    results: list[dict[str, Any]] = []
    ids_list = result.get("ids", [[]])
    docs_list = result.get("documents", [[]])
    mds_list = result.get("metadatas", [[]])
    dists_list = result.get("distances", [[]])

    ids = ids_list[0] if ids_list else []
    docs = docs_list[0] if docs_list else []
    mds = mds_list[0] if mds_list else []
    dists = dists_list[0] if dists_list else []

    for i in range(len(ids)):
        results.append({
            "id": ids[i] if i < len(ids) else "",
            "content": docs[i] if i < len(docs) else "",
            "metadata": mds[i] if i < len(mds) else {},
            "distance": dists[i] if i < len(dists) else 0.0,
        })

    logger.info(
        "retrieve_cross_modal user=%s modality=%s returned=%d",
        user_id, modality, len(results),
    )
    return results


async def retrieve_combined(
    user_id: str,
    query: str,
    k: int = 5,
) -> dict[str, Any]:
    """组合检索：L0 + L1 + 因果链。"""
    l0 = await retrieve_l0_memories(user_id)
    l1 = await retrieve_l1_memories(user_id, query, k=k)

    # 对 L0 记忆尝试构建因果链
    causal: list[dict[str, Any]] = []
    for mem in l0[:3]:  # 只对前 3 条 L0 记忆构建因果链，避免过多查询
        chain = await retrieve_causal_chain(user_id, mem["id"])
        if chain:
            causal.extend(chain)

    logger.info(
        "retrieve_combined user=%s l0=%d l1=%d causal=%d",
        user_id, len(l0), len(l1), len(causal),
    )
    return {"l0": l0, "l1": l1, "causal": causal}