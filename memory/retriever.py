"""记忆检索：
- L0：MySQL 全量加载核心记忆
- L1：weaviate 向量检索 Top-K
- 因果链：MySQL memory_relations
- 多模态：weaviate 跨模态检索（文本 query → 图片/音频/视频 embedding 命中）
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from config.config import get_settings
from utils.request_context import log_stage, merge_extra

logger = logging.getLogger(__name__)


def _log_table_error(op: str, exc: Exception, table: str, *, extra: dict | None = None) -> None:
    """把 MySQL 异常翻译成可读的提示，避免被 2013 'Lost connection' 误导。

    - 1146: 表不存在（通常是被人工 DROP 了，init_schema() 会自动补建）
    - 2013/2006: 真断连（网络 / 服务重启 / max_connections 触顶）
    - 其它: 原样
    """
    code = None
    args = getattr(exc, "args", ())
    if args and isinstance(args[0], int):
        code = args[0]
    fields = {"op": op, "table": table, "code": code, "exc": str(exc)[:200], **(extra or {})}
    if code == 1146:
        logger.warning(
            "%s: 表 %s 不存在 — 服务启动时会通过 init_schema() 幂等创建；若持续缺失请检查 init_schema 日志",
            op, table,
            extra=merge_extra(stage="db", event="table_missing", **fields),
        )
    elif code in (2013, 2006, 2014, 2055):
        logger.warning(
            "%s: MySQL 断连 (%s) — 检查网络 / 服务状态 / max_connections: %s",
            op, code, exc,
            extra=merge_extra(stage="db", event="lost_connection", **fields),
        )
    else:
        logger.warning(
            "%s failed: %s", op, exc,
            extra=merge_extra(stage="db", event="error", **fields),
        )


# ---------- L0 ----------

async def load_l0_memories(user_id: str, limit: int | None = None) -> list[dict]:
    """从 memories 表加载用户的 L0 长期核心记忆。"""
    if not user_id:
        return []
    settings = get_settings().memory
    limit = int(limit or settings.l0_limit)
    t0 = time.perf_counter()
    try:
        from database import fetch_all

        rows = await fetch_all(
            """
            SELECT id, content, memory_type AS category, vector_id,
                   created_at, updated_at
            FROM memories
            WHERE user_id=%s AND level='L0'
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user_id, max(1, limit)),
        )
        out = [
            {
                "id": r["id"],
                "content": r["content"],
                "category": r.get("category"),
                "vector_id": r.get("vector_id"),
            }
            for r in rows
        ]
        logger.info(
            "load_l0_memories ok",
            extra=merge_extra(
                stage="memory_l0",
                event="ok",
                user_id=user_id,
                count=len(out),
                limit=limit,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return out
    except Exception as e:
        _log_table_error("load_l0_memories", e, "memories", extra={"user_id": user_id})
        return []


# ---------- L1 ----------

async def load_l1_summaries(user_id: str, limit: int | None = None) -> list[dict]:
    """从 MySQL 加载最近 L1/L2 摘要。"""
    if not user_id:
        return []
    settings = get_settings().memory
    limit = int(limit or settings.l1_topk)
    t0 = time.perf_counter()
    try:
        from database import fetch_all
        from llm.cascade import _strip_think

        rows = await fetch_all(
            """
            SELECT id, level, content, summary, emotion_tag, emotion_intensity, created_at
            FROM memories
            WHERE user_id=%s AND level IN ('L1','L2')
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (user_id, max(1, limit)),
        )
        out = []
        for r in rows:
            raw = r.get("summary") or r.get("content") or ""
            text = _strip_think(raw)
            out.append(
                {
                    "id": r["id"],
                    "level": r["level"],
                    "text": text,
                    "emotion_tag": r["emotion_tag"],
                    "emotion_intensity": float(r["emotion_intensity"] or 0.0),
                }
            )
        logger.info(
            "load_l1_summaries ok",
            extra=merge_extra(
                stage="memory_l1",
                event="ok",
                user_id=user_id,
                count=len(out),
                limit=limit,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return out
    except Exception as e:
        _log_table_error("load_l1_summaries", e, "memories", extra={"user_id": user_id})
        return []


# ---------- 因果链 ----------

async def causal_chain(user_id: str, memory_id: int, depth: int = 2) -> list[dict]:
    """从 MySQL memory_relations 查询以 memory_id 为起点的因果链（兼容旧 schema）。"""
    if not user_id or not memory_id:
        return []
    t0 = time.perf_counter()
    try:
        from database import fetch_all

        rows = await fetch_all(
            """
            SELECT source_id, target_id,
                   COALESCE(NULLIF(relation_type, ''), relation) AS relation,
                   COALESCE(weight, confidence, 1.0) AS weight
            FROM memory_relations
            WHERE user_id=%s AND (source_id=%s OR target_id=%s)
            ORDER BY weight DESC, created_at DESC
            LIMIT %s
            """,
            (user_id, memory_id, memory_id, max(1, depth * 4)),
        )
        out = [
            {
                "source_id": r["source_id"],
                "target_id": r["target_id"],
                "relation": r["relation"],
                "weight": float(r["weight"] or 1.0),
            }
            for r in rows
        ]
        logger.info(
            "causal_chain ok",
            extra=merge_extra(
                stage="memory_causal",
                event="ok",
                user_id=user_id,
                memory_id=memory_id,
                depth=depth,
                count=len(out),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return out
    except Exception as e:
        _log_table_error("causal_chain", e, "memory_relations", extra={"user_id": user_id})
        return []


# ---------- 多模态（weaviate 跨模态） ----------

async def multimodal_search(
    user_id: str,
    query: str,
    *,
    query_embedding: list[float] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """weaviate 跨模态检索：文本 query → 命中图片/音频/视频/文本。

    实现说明：
    - EchoDoc 存的是 Chinese-CLIP 512 维向量（文本/图像共用空间）。
    - 必须用 **CLIP 文本编码器** 编码 query 才能在同空间下做相似检索。
    - 旧实现错用 BGE-M3（1024 维），维度不匹配被 Weaviate 拒，命中永远为空。
    """
    settings = get_settings().memory
    top_k = int(top_k or settings.l1_topk)
    # UI 默认只看最相关的少数条；取 min(top_k, search_max_hits) 兜底
    max_hits = max(1, int(getattr(settings, "search_max_hits", 1) or 1))
    top_k = min(top_k, max_hits)
    if not query or not user_id:
        return {"hits": [], "modality_counts": {}}
    t0 = time.perf_counter()
    try:
        if query_embedding is None:
            from embedding.models import compute_text_embeddings

            vecs = await asyncio.to_thread(compute_text_embeddings, [query])
            query_embedding = vecs[0] if vecs else []
        if not query_embedding:
            return {"hits": [], "modality_counts": {}}
        from vector.vector_store import get_vector_store

        vs = await asyncio.to_thread(get_vector_store)

        def _fixed_embed(_texts: list[str]) -> list[list[float]]:
            return [query_embedding]

        result = await asyncio.to_thread(
            vs.query,
            query,
            top_k,
            _fixed_embed,
            {"userId": user_id},
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        mds = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits = []
        modality_counts: dict[str, int] = {}
        score_dist: list[float] = []
        for i, doc in enumerate(docs):
            md = mds[i] if i < len(mds) else {}
            # 推断 modality：优先读显式字段；否则按 chunkIndex/totalChunks 判定（图像均为单 chunk）
            modality = (
                (md or {}).get("modality")
                or (md or {}).get("source")
                or ("image" if (md or {}).get("sourceUrl") and (md or {}).get("totalChunks", 1) == 1 else "text")
            )
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            sim = 1.0 - float(distances[i]) if i < len(distances) else 0.0
            score_dist.append(round(sim, 4))
            hits.append(
                {
                    "id": ids[i] if i < len(ids) else "",
                    "content": doc,
                    "metadata": md,
                    "modality": modality,
                    "similarity": sim,
                }
            )
        top_score = max(score_dist) if score_dist else 0.0
        # 按相似度倒序，截到 max_hits 条；保证只返回最相关的一条
        hits.sort(key=lambda h: h.get("similarity", 0.0), reverse=True)
        hits = hits[:max_hits]
        logger.info(
            "multimodal_search ok",
            extra=merge_extra(
                stage="memory_multimodal",
                event="ok",
                user_id=user_id,
                query_preview=(query or "")[:80],
                top_k=top_k,
                max_hits=max_hits,
                hit_count=len(hits),
                modality_counts=modality_counts,
                top_score=top_score,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return {"hits": hits, "modality_counts": modality_counts}
    except Exception as e:
        logger.warning(
            "multimodal_search failed",
            extra=merge_extra(
                stage="memory_multimodal",
                event="error",
                user_id=user_id,
                error=str(e)[:200],
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return {"hits": [], "modality_counts": {}}


# ---------- Persona ----------

async def load_persona(user_id: str) -> str:
    """加载用户人格；缺省时使用 config.prompts.DEFAULT_PERSONA。"""
    from config.prompts import DEFAULT_PERSONA

    if not user_id:
        return DEFAULT_PERSONA
    t0 = time.perf_counter()
    try:
        from database import fetch_one

        row = await fetch_one("SELECT persona FROM personas WHERE user_id=%s", (user_id,))
        if row and row.get("persona"):
            logger.info(
                "load_persona hit",
                extra=merge_extra(
                    stage="persona",
                    event="hit",
                    user_id=user_id,
                    len=len(row["persona"]),
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                ),
            )
            return row["persona"]
        logger.info(
            "load_persona fallback",
            extra=merge_extra(
                stage="persona",
                event="fallback_default",
                user_id=user_id,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
    except Exception as e:
        _log_table_error("load_persona", e, "personas", extra={"user_id": user_id})
    return DEFAULT_PERSONA


async def save_persona(user_id: str, persona: str) -> None:
    if not user_id or not persona:
        return
    t0 = time.perf_counter()
    try:
        from database import execute

        await execute(
            """
            INSERT INTO personas (user_id, persona)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE persona=VALUES(persona)
            """,
            (user_id, persona),
        )
        logger.info(
            "save_persona ok",
            extra=merge_extra(
                stage="persona",
                event="saved",
                user_id=user_id,
                len=len(persona),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
    except Exception as e:
        _log_table_error("save_persona", e, "personas", extra={"user_id": user_id})


# ---------- 组合：构建 chat 上下文 ----------

async def build_chat_context(user_id: str, query: str, *, enable_multimodal: bool = True) -> dict:
    """构建 chat 接口所需的注入上下文：
    - persona
    - L0 l0_memories（list[str]，从 memories 表 level='L0' 读取）
    - L1 recent_summaries（list[str]）
    - l1_hits（最近相关记忆，仅当 enable_multimodal=True 时从 EchoDoc 跨模态检索）

    `enable_multimodal=False` 时跳过 multimodal_search，避免闲聊/回忆类查询白白触发
    Weaviate EchoDoc 跨模态检索（200~800ms，偶发 9s+）。
    """
    with log_stage(logger, "build_chat_context", start_msg="build_chat_context begin", level=logging.INFO) as meta:
        if enable_multimodal:
            persona, l0, recent, l1 = await asyncio.gather(
                load_persona(user_id),
                load_l0_memories(user_id),
                load_l1_summaries(user_id),
                multimodal_search(user_id, query, top_k=get_settings().memory.l1_topk),
            )
        else:
            persona, l0, recent = await asyncio.gather(
                load_persona(user_id),
                load_l0_memories(user_id),
                load_l1_summaries(user_id),
            )
            l1 = {"hits": [], "modality_counts": {}}
        meta.update(
            enable_multimodal=enable_multimodal,
            l1_hit_count=len(l1.get("hits", []) or []),
        )
    return {
        "persona": persona,
        "l0_memories": [m["content"] for m in l0],
        "recent_summaries": [
            f"[{m['level']}] {m['text']}" for m in recent
        ],
        "l1_hits": l1.get("hits", []),
    }