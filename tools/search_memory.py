"""search_memory：进程内检索 weaviate（记忆 + RAG 文档库）+ MySQL。

- EchoMemory（BGE-M3 768 维）：对话记忆、L1 摘要
- EchoDoc（Chinese-CLIP 512 维）：用户上传的图片与文本片段（RAG 库）
- 因果链：MySQL `memory_relations` 扩展

设计要点：旧的实现只查 EchoMemory 且用 BGE-M3 编码，导致用户问"在 RAG 库找图"时永远没有命中。
本工具并行查两个 collection，并在结果中标记 modality，让上层 LLM 知道哪些是图片。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from tools.base import BaseTool, ToolResult, ok
from utils.request_context import log_exception

logger = logging.getLogger(__name__)


class SearchMemoryTool(BaseTool):
    name = "search_memory"
    description = (
        "search_memory(query: str, user_id: str, top_k: int = 8) -> "
        "在长期记忆与用户的 RAG 文档库中检索与 query 最相关的若干条，"
        "返回 {hits: [{content, similarity, modality, ...}], causal: [...]}。"
        "modality 取值 'memory' | 'text' | 'image' | 'audio' | 'video'。"
        "当对话涉及过去的事件、偏好、关系，或希望查找用户上传过的图片/文档片段时使用。"
    )

    async def arun(
        self,
        query: str = "",
        user_id: str = "",
        top_k: int = 8,
        role_id: str = "default",
        **_,
    ) -> ToolResult:
        return await _search_memory_async(query=query, user_id=user_id, top_k=top_k, role_id=role_id)

    def run(
        self,
        query: str = "",
        user_id: str = "",
        top_k: int = 8,
        role_id: str = "default",
        **_,
    ) -> ToolResult:
        try:
            asyncio.get_running_loop()
            from tools.base import fail

            return fail("Use arun() in async context")
        except RuntimeError:
            return asyncio.run(
                _search_memory_async(query=query, user_id=user_id, top_k=top_k, role_id=role_id)
            )


async def _search_memory_async(*, query: str, user_id: str, top_k: int, role_id: str = "default") -> ToolResult:
    from config.config import get_settings

    top_k = max(1, min(int(top_k or 8), 50))
    if not query or not user_id:
        return ok({"hits": [], "causal": []})

    settings = get_settings().memory
    # UI 默认只看最相关的少数条；最终取 min(top_k, search_max_hits)
    max_hits = max(1, int(getattr(settings, "search_max_hits", 1) or 1))
    top_k = min(top_k, max_hits)
    hits: list[dict[str, Any]] = []

    # ---------- 并行：EchoMemory (BGE-M3) + EchoDoc (CLIP) ----------
    async def _search_memory_store() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            from vector.memory_store import get_memory_vector_store
            from embedding.bge_m3 import embed_texts

            q_vec_list = await asyncio.to_thread(embed_texts, [query])
            q_vec = q_vec_list[0] if q_vec_list else []
            if not q_vec:
                return out
            vs = await asyncio.to_thread(get_memory_vector_store)

            def _fixed_embed(_texts: list[str]) -> list[list[float]]:
                return [q_vec]

            result = await asyncio.to_thread(
                vs.query,
                query,
                top_k,
                _fixed_embed,
                {"userId": user_id, "roleId": role_id},
            )
            ids = result.get("ids", [[]])[0]
            docs = result.get("documents", [[]])[0]
            mds = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            for i, d in enumerate(docs):
                md = mds[i] if i < len(mds) else {}
                out.append(
                    {
                        "id": ids[i] if i < len(ids) else "",
                        "content": d,
                        "metadata": md,
                        "similarity": 1.0 - float(distances[i]) if i < len(distances) else 0.0,
                        "source": "EchoMemory",
                        "modality": (md or {}).get("modality") or "memory",
                    }
                )
        except Exception as e:
            log_exception(
                logger,
                "EchoMemory search failed",
                exc=e,
                level=logging.WARNING,
                stage="search_memory",
                event="memory_store_error",
                user_id=user_id,
                query_preview=query[:80],
                top_k=top_k,
            )
        return out

    async def _search_doc_store() -> list[dict[str, Any]]:
        """EchoDoc：CLIP 文本 query → 命中图片/文本/音频。"""
        out: list[dict[str, Any]] = []
        try:
            from vector.vector_store import get_vector_store
            from embedding.models import compute_text_embeddings

            q_vec_list = await asyncio.to_thread(compute_text_embeddings, [query])
            q_vec = q_vec_list[0] if q_vec_list else []
            if not q_vec:
                return out
            vs = await asyncio.to_thread(get_vector_store)

            def _fixed_embed(_texts: list[str]) -> list[list[float]]:
                return [q_vec]

            result = await asyncio.to_thread(
                vs.query,
                query,
                top_k,
                _fixed_embed,
                {"userId": user_id, "roleId": role_id},
            )
            ids = result.get("ids", [[]])[0]
            docs = result.get("documents", [[]])[0]
            mds = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            for i, d in enumerate(docs):
                md = mds[i] if i < len(mds) else {}
                # 推断 modality：显式字段 → sourceUrl 形态 → 兜底
                modality = (
                    (md or {}).get("modality")
                    or ("image" if (md or {}).get("sourceUrl") and (md or {}).get("totalChunks", 1) == 1 else "text")
                )
                out.append(
                    {
                        "id": ids[i] if i < len(ids) else "",
                        "content": d,
                        "metadata": md,
                        "similarity": 1.0 - float(distances[i]) if i < len(distances) else 0.0,
                        "source": "EchoDoc",
                        "modality": modality,
                    }
                )
        except Exception as e:
            log_exception(
                logger,
                "EchoDoc search failed",
                exc=e,
                level=logging.WARNING,
                stage="search_memory",
                event="doc_store_error",
                user_id=user_id,
                query_preview=query[:80],
                top_k=top_k,
            )
        return out

    memory_hits, doc_hits = await asyncio.gather(_search_memory_store(), _search_doc_store())
    hits = memory_hits + doc_hits
    # 按相似度倒序，截断到 top_k（用户希望只看到最相关的一条）
    hits.sort(key=lambda h: h.get("similarity", 0.0), reverse=True)
    hits = hits[: max(1, top_k)]

    # ---------- 因果链：MySQL memory_relations（仅对 memory 类命中） ----------
    causal: list[dict] = []
    try:
        from database import fetch_all

        memory_hits_only = [h for h in hits if h.get("source") == "EchoMemory"][:3]
        for h in memory_hits_only:
            content = h.get("content") or ""
            if not content:
                continue
            rows = await fetch_all(
                """
                SELECT id, level, content, emotion_tag, emotion_intensity
                FROM memories
                WHERE user_id=%s AND role_id=%s AND content LIKE %s
                ORDER BY created_at DESC LIMIT 5
                """,
                (user_id, role_id, f"%{content[:32]}%"),
            )
            for r in rows:
                causal.append(
                    {
                        "memory_id": r["id"],
                        "level": r["level"],
                        "content": r["content"],
                        "emotion_tag": r["emotion_tag"],
                        "emotion_intensity": float(r["emotion_intensity"] or 0.0),
                    }
                )
        causal = causal[: settings.l1_topk]
    except Exception as e:
        log_exception(
            logger,
            "causal chain fetch failed",
            exc=e,
            level=logging.WARNING,
            stage="search_memory",
            event="causal_error",
            user_id=user_id,
            memory_hits=len(memory_hits_only),
            total_hits=len(hits),
        )

    # 统计 modality 分布，便于上层观察
    modality_counts: dict[str, int] = {}
    for h in hits:
        m = h.get("modality") or "unknown"
        modality_counts[m] = modality_counts.get(m, 0) + 1

    return ok({"hits": hits, "causal": causal, "modality_counts": modality_counts})
