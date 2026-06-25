from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from database.mysql import get_session
from database.models import CoreMemory
from embedding.models import compute_text_embeddings
from vector.vector_store import get_vector_store

logger = logging.getLogger(__name__)


async def search_memory(user_id: str, query: str, k: int = 5) -> dict[str, Any]:
    """搜索用户记忆：向量检索 + L0 核心记忆。

    Args:
        user_id: 用户 ID（租户隔离）。
        query: 自然语言查询。
        k: 返回结果数量。

    Returns:
        {"vector_results": [...], "core_memories": [...]}
    """
    vector_results: list[dict[str, Any]] = []
    core_memories: list[dict[str, Any]] = []

    # ── 向量检索 ──────────────────────────────────────────────────
    try:
        vs = get_vector_store()
        result = vs.query(
            query_text=query,
            n_results=k,
            embedding_fn=compute_text_embeddings,
            where={"userId": user_id},
        )
        docs = result.get("documents", [[]])[0] or []
        mds = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []
        ids = result.get("ids", [[]])[0] or []

        for i in range(len(docs)):
            vector_results.append({
                "id": ids[i] if i < len(ids) else "",
                "text": docs[i] if i < len(docs) else "",
                "metadata": mds[i] if i < len(mds) else {},
                "distance": distances[i] if i < len(distances) else 0.0,
            })
    except Exception as e:
        logger.error("Vector search failed for user=%s: %s", user_id, e)

    # ── MySQL L0 核心记忆 ─────────────────────────────────────────
    try:
        async with get_session() as session:
            stmt = (
                select(CoreMemory)
                .where(CoreMemory.user_id == user_id, CoreMemory.layer == "L0")
                .order_by(CoreMemory.importance.desc())
                .limit(k)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            for row in rows:
                core_memories.append({
                    "id": row.id,
                    "content": row.content,
                    "memory_type": row.memory_type,
                    "importance": row.importance,
                    "emotion_tag": row.emotion_tag,
                    "emotion_intensity": row.emotion_intensity,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                })
    except Exception as e:
        logger.error("MySQL core memory query failed for user=%s: %s", user_id, e)

    return {"vector_results": vector_results, "core_memories": core_memories}


def get_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "搜索用户记忆，包括向量检索和 L0 核心记忆。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "用户 ID，用于租户隔离",
                    },
                    "query": {
                        "type": "string",
                        "description": "自然语言搜索查询",
                    },
                    "k": {
                        "type": "integer",
                        "description": "返回结果数量，默认 5",
                    },
                },
                "required": ["user_id", "query"],
            },
        },
    }