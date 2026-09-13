"""清空对话记忆链路 (方案C: 7 项)。

按用户授权: 不备份,直接 DELETE。
执行内容:
  - memories WHERE level IN ('L0','L1','L2')
  - memory_relations
  - memory_extract_logs
  - emotion_logs
  - chat_sessions
  - chat_messages
  - Weaviate EchoMemory (整 collection 删 + 重建,保证空)
"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")


async def wipe_mysql() -> dict[str, int]:
    from database import execute, fetch_all

    affected = {}
    # 1) memories: 按 level 限定。level 列 NOT NULL DEFAULT 'L1', 实际全是 L0/L1/L2,
    #    但显式带 WHERE 防御性。
    r = await fetch_all("SELECT COUNT(*) AS c FROM memories")
    affected["memories_before"] = int(r[0]["c"])
    await execute("DELETE FROM memories WHERE level IN ('L0','L1','L2')")
    r = await fetch_all("SELECT COUNT(*) AS c FROM memories")
    affected["memories_after"] = int(r[0]["c"])

    # 2) memory_relations
    r = await fetch_all("SELECT COUNT(*) AS c FROM memory_relations")
    affected["memory_relations_before"] = int(r[0]["c"])
    await execute("DELETE FROM memory_relations")
    r = await fetch_all("SELECT COUNT(*) AS c FROM memory_relations")
    affected["memory_relations_after"] = int(r[0]["c"])

    # 3) memory_extract_logs
    r = await fetch_all("SELECT COUNT(*) AS c FROM memory_extract_logs")
    affected["memory_extract_logs_before"] = int(r[0]["c"])
    await execute("DELETE FROM memory_extract_logs")
    r = await fetch_all("SELECT COUNT(*) AS c FROM memory_extract_logs")
    affected["memory_extract_logs_after"] = int(r[0]["c"])

    # 4) emotion_logs
    r = await fetch_all("SELECT COUNT(*) AS c FROM emotion_logs")
    affected["emotion_logs_before"] = int(r[0]["c"])
    await execute("DELETE FROM emotion_logs")
    r = await fetch_all("SELECT COUNT(*) AS c FROM emotion_logs")
    affected["emotion_logs_after"] = int(r[0]["c"])

    # 5) chat_sessions
    r = await fetch_all("SELECT COUNT(*) AS c FROM chat_sessions")
    affected["chat_sessions_before"] = int(r[0]["c"])
    await execute("DELETE FROM chat_sessions")
    r = await fetch_all("SELECT COUNT(*) AS c FROM chat_sessions")
    affected["chat_sessions_after"] = int(r[0]["c"])

    # 6) chat_messages
    r = await fetch_all("SELECT COUNT(*) AS c FROM chat_messages")
    affected["chat_messages_before"] = int(r[0]["c"])
    await execute("DELETE FROM chat_messages")
    r = await fetch_all("SELECT COUNT(*) AS c FROM chat_messages")
    affected["chat_messages_after"] = int(r[0]["c"])

    return affected


async def wipe_weaviate() -> dict[str, int]:
    """整 collection 删除并重建。MemoryVectorStore 单例会在下次访问时自动重建。"""
    import httpx
    from config.config import get_settings
    from vector.vector_store import _WeaviateHttpClient

    settings = get_settings().weaviate
    base = settings.resolved_url()
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    affected = {}
    # 删前计数
    agg = '{"query":"{ Aggregate { EchoMemory { meta { count } } } }"}'
    async with httpx.AsyncClient(base_url=base, headers=headers, timeout=15.0) as c:
        r = await c.post("/v1/graphql", json=agg)
        data = r.json().get("data", {}).get("Aggregate", {}).get("EchoMemory", [{}])
        before = int(data[0].get("meta", {}).get("count", 0)) if data else 0
        affected["echo_memory_before"] = before
        # 整 class 删除
        r = await c.delete("/v1/schema/EchoMemory")
        # 200/204 = OK; 404 = 不存在(也算成功);其它抛错
        if r.status_code not in (200, 204, 404):
            raise RuntimeError(f"EchoMemory delete failed: {r.status_code} {r.text}")
        affected["echo_memory_after_drop"] = 0

    # 触发 MemoryVectorStore 重建 schema(单例已被 drop,重置)
    from vector import memory_store

    memory_store.reset_memory_vector_store()
    await asyncio.to_thread(memory_store.get_memory_vector_store)

    # 再统计确认
    async with httpx.AsyncClient(base_url=base, headers=headers, timeout=15.0) as c:
        r = await c.post("/v1/graphql", json=agg)
        data = r.json().get("data", {}).get("Aggregate", {}).get("EchoMemory", [{}])
        after = int(data[0].get("meta", {}).get("count", 0)) if data else 0
        affected["echo_memory_after_recreate"] = after
    return affected


async def main() -> None:
    print("=== Wiping 7 items per Plan C ===")
    mysql_result = await wipe_mysql()
    for k, v in mysql_result.items():
        print(f"  MySQL {k}: {v}")
    print()
    weaviate_result = await wipe_weaviate()
    for k, v in weaviate_result.items():
        print(f"  Weaviate {k}: {v}")
    print()
    print("=== Done. All 7 items cleared. ===")


if __name__ == "__main__":
    asyncio.run(main())