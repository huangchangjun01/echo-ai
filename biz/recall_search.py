"""回忆记忆检索：对话消息 → EchoRecall top5 摘要 → LLM 生成回复。

`search_recall_for_chat` 给 chat 模块一个统一入口：
- 输入：user_id, role_id, message
- 输出：top5 摘要 + 对应 md_key + similarity 列表
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from embedding import bge_m3
from utils.request_context import log_exception, merge_extra
from vector.recall_store import get_recall_store

logger = logging.getLogger(__name__)


async def search_recall_for_chat(
    user_id: str,
    role_id: str,
    message: str,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """根据用户消息在 EchoRecall 中检索 top_k 摘要。

    返回元素结构：{memoryId, mdKey, topic, summary, similarity}
    若 EchoRecall 不可用或消息为空，返回 []。
    """
    if not message.strip():
        return []
    try:
        # 1) 先用 BGE-M3 把用户消息向量化（一次性）
        vecs = await asyncio.to_thread(bge_m3.embed_texts, [message])
        if not vecs:
            return []
        q_vec = list(map(float, vecs[0]))

        # 2) 直接走底层 graphql 复用 q_vec（避免 query() 二次 embed）
        from vector.recall_store import get_recall_store as _get
        store = _get()
        items = store._query_with_vector(q_vec, n_results=top_k, where={"userId": user_id, "roleId": role_id or "default"})
        out: list[dict[str, Any]] = []
        for it in items:
            md = it.get("_meta") or {}
            out.append(
                {
                    "memoryId": md.get("memoryId") or it.get("id", ""),
                    "mdKey": md.get("mdKey"),
                    "topic": md.get("topic"),
                    "summary": it.get("summary", ""),
                    "similarity": it.get("similarity", 0.0),
                }
            )
        logger.info(
            "recall search ok",
            extra=merge_extra(
                stage="recall_search",
                event="ok",
                user_id=user_id,
                role_id=role_id,
                top_k=top_k,
                hits=len(out),
            ),
        )
        return out
    except Exception as e:
        log_exception(
            logger,
            "recall search failed (fallback empty)",
            exc=e,
            level=logging.WARNING,
            stage="recall_search",
            event="error",
            user_id=user_id,
        )
        return []


async def fetch_memory_md(md_key: str) -> str | None:
    """按 md_key 经 echo-core 代理端点拿 md 原文（追问细节场景，绕开 Qiniu CDN 421）。"""
    import logging
    logger_local = logging.getLogger(__name__)
    parts = md_key.split("/")
    if len(parts) < 4 or parts[0] != "memory":
        return None
    try:
        import os
        import httpx

        from utils.request_context import log_exception

        user_id = parts[1]
        memory_id = parts[-1].replace(".md", "")
        base = (os.environ.get("ECHO_CORE_BASE_URL") or "http://localhost:8080").rstrip("/")
        token = os.environ.get("ECHO_CORE_INTERNAL_TOKEN") or ""
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Internal-Token"] = token
        async with httpx.AsyncClient(timeout=60, headers=headers) as c:
            r = await c.post(f"{base}/api/memory/md-content", json={"userId": user_id, "memoryId": memory_id})
            if r.status_code != 200:
                logger_local.warning("fetch_memory_md: echo-core returned %d", r.status_code)
                return None
            return r.content.decode("utf-8", errors="replace")
    except Exception as e:
        log_exception(
            logger,
            "fetch_memory_md failed",
            exc=e,
            level=logging.WARNING,
            stage="recall_fetch_md",
            event="error",
            md_key=md_key,
        )
        return None