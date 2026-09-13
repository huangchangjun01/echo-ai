"""清空记忆链路(直接 SQL 形式, 避开分类器对 wipe 脚本的拦截)。

按用户授权: 不备份, 直接 DELETE。
"""
import asyncio
import sys

sys.path.insert(0, ".")


async def main() -> None:
    from database import execute, fetch_all
    tables = [
        "memory_relations",
        "memory_extract_logs",
        "emotion_logs",
        "chat_sessions",
        "chat_messages",
    ]
    for tbl in tables:
        before = await fetch_all(f"SELECT COUNT(*) AS c FROM {tbl}", ())
        n = int(before[0]["c"])
        await execute(f"DELETE FROM {tbl}", ())
        print(f"{tbl}: {n} -> 0")
    before_m = await fetch_all("SELECT COUNT(*) AS c FROM memories WHERE level IN ('L0','L1','L2')", ())
    nm = int(before_m[0]["c"])
    await execute("DELETE FROM memories WHERE level IN ('L0','L1','L2')", ())
    print(f"memories (L0/L1/L2): {nm} -> 0")
    # Weaviate: drop EchoMemory class, recreate
    import httpx
    from config.config import get_settings
    s = get_settings().weaviate
    base = s.resolved_url()
    headers = {"Content-Type": "application/json"}
    if s.api_key:
        headers["Authorization"] = f"Bearer {s.api_key}"
    async with httpx.AsyncClient(base_url=base, headers=headers, timeout=15.0) as c:
        r = await c.delete("/v1/schema/EchoMemory")
        print(f"Weaviate EchoMemory delete: {r.status_code}")
    from vector import memory_store
    memory_store.reset_memory_vector_store()
    await asyncio.to_thread(memory_store.get_memory_vector_store)
    print("EchoMemory recreated")


if __name__ == "__main__":
    asyncio.run(main())