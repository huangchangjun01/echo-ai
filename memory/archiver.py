from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select, func, delete

from config.config import get_settings
from database.models import CoreMemory, MemoryRelation, MemorySummary
from database.mysql import get_session

logger = logging.getLogger(__name__)


async def archive_memory(
    user_id: str,
    content: str,
    memory_type: str,
    importance: float,
    emotion_tag: str | None,
    emotion_intensity: float,
    weaviate_uuid: str | None,
    embedding_dim: int,
    summary: str | None,
) -> int:
    """归档一条记忆，根据 importance 判定层级并写入 core_memories 表。

    层级判定：
    - importance >= L0_MIN_IMPORTANCE → L0
    - importance >= L1_MIN_IMPORTANCE → L1
    - 否则 → L2
    """
    settings = get_settings().memory

    if importance >= settings.l0_min_importance:
        layer = "L0"
        max_count = settings.l0_max_count
    elif importance >= settings.l1_min_importance:
        layer = "L1"
        max_count = settings.l1_max_count
    else:
        layer = "L2"
        max_count = settings.l2_max_count

    async with get_session() as session:
        # 检查同级记忆数量是否超限
        count_result = await session.execute(
            select(func.count(CoreMemory.id)).where(
                CoreMemory.user_id == user_id,
                CoreMemory.layer == layer,
            )
        )
        current_count = count_result.scalar() or 0

        if current_count >= max_count:
            # 超限时删除最旧的低重要性记忆
            excess = current_count - max_count + 1
            subquery = (
                select(CoreMemory.id)
                .where(
                    CoreMemory.user_id == user_id,
                    CoreMemory.layer == layer,
                )
                .order_by(CoreMemory.importance.asc(), CoreMemory.created_at.asc())
                .limit(excess)
            ).subquery()
            await session.execute(
                delete(CoreMemory).where(CoreMemory.id.in_(select(subquery.c.id)))
            )
            logger.info(
                "archived %d old memories from layer=%s user=%s",
                excess, layer, user_id,
            )

        memory = CoreMemory(
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            layer=layer,
            importance=importance,
            emotion_tag=emotion_tag,
            emotion_intensity=emotion_intensity,
            weaviate_uuid=weaviate_uuid,
            embedding_dim=embedding_dim,
            summary=summary,
        )
        session.add(memory)
        await session.flush()
        memory_id = memory.id
        logger.info(
            "archived memory id=%s layer=%s importance=%.3f user=%s",
            memory_id, layer, importance, user_id,
        )

    return memory_id


async def archive_relation(
    source_id: int,
    target_id: int,
    relation_type: str,
    confidence: float,
) -> int:
    """写入一条记忆关系到 memory_relations 表。"""
    async with get_session() as session:
        relation = MemoryRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
        )
        session.add(relation)
        await session.flush()
        relation_id = relation.id
        logger.info(
            "archived relation id=%s type=%s source=%s target=%s confidence=%.3f",
            relation_id, relation_type, source_id, target_id, confidence,
        )

    return relation_id


async def archive_summary(user_id: str, summary: str, memory_ids: str) -> int:
    """写入记忆摘要到 memory_summaries 表，超限时删除最旧的。"""
    settings = get_settings().memory

    async with get_session() as session:
        count_result = await session.execute(
            select(func.count(MemorySummary.id)).where(
                MemorySummary.user_id == user_id,
            )
        )
        current_count = count_result.scalar() or 0

        if current_count >= settings.max_summaries:
            excess = current_count - settings.max_summaries + 1
            subquery = (
                select(MemorySummary.id)
                .where(MemorySummary.user_id == user_id)
                .order_by(MemorySummary.created_at.asc())
                .limit(excess)
            ).subquery()
            await session.execute(
                delete(MemorySummary).where(MemorySummary.id.in_(select(subquery.c.id)))
            )
            logger.info("deleted %d old summaries for user=%s", excess, user_id)

        record = MemorySummary(
            user_id=user_id,
            summary=summary,
            memory_ids=memory_ids,
        )
        session.add(record)
        await session.flush()
        summary_id = record.id
        logger.info("archived summary id=%s user=%s", summary_id, user_id)

    return summary_id


async def get_l0_memories(user_id: str) -> list[dict[str, Any]]:
    """查询 user_id 的 L0 层核心记忆，返回 dict 列表。"""
    async with get_session() as session:
        result = await session.execute(
            select(CoreMemory)
            .where(CoreMemory.user_id == user_id, CoreMemory.layer == "L0")
            .order_by(CoreMemory.importance.desc(), CoreMemory.created_at.desc())
        )
        rows = result.scalars().all()
        memories = [
            {
                "id": row.id,
                "user_id": row.user_id,
                "content": row.content,
                "memory_type": row.memory_type,
                "layer": row.layer,
                "importance": row.importance,
                "emotion_tag": row.emotion_tag,
                "emotion_intensity": row.emotion_intensity,
                "weaviate_uuid": row.weaviate_uuid,
                "embedding_dim": row.embedding_dim,
                "summary": row.summary,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ]
        logger.info("get_l0_memories user=%s count=%d", user_id, len(memories))
    return memories