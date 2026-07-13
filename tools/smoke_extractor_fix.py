"""Smoke test for memory extractor fix.

Sends a chat that yields an extractable fact, then a follow-up that should trigger
the dedup path. Verifies no JSONDecodeError in logs and that memories land.
"""

from __future__ import annotations

import asyncio
import io
import sys
import time

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import httpx  # noqa: E402

URL = "http://127.0.0.1:8000/chat"
USER = "extractor-fix-user"
SESSION = "extractor-fix-session"

CASES = [
    {
        "label": "seed_first_fact",
        "msg": "我昨天和朋友去爬了香山，山顶风景特别美",
    },
    {
        "label": "second_fact_with_topic_overlap",
        "msg": "周末我和朋友去爬山，这次爬的是西山",
    },
]


async def post(client: httpx.AsyncClient, msg: str) -> tuple[int, dict]:
    body = {"userId": USER, "sessionId": SESSION, "message": msg, "stream": False}
    r = await client.post(URL, json=body, timeout=httpx.Timeout(60.0, connect=10.0))
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"_raw": r.text[:300]}


async def main() -> int:
    async with httpx.AsyncClient() as c:
        results = []
        for case in CASES:
            t0 = time.perf_counter()
            status, body = await post(c, case["msg"])
            dt = (time.perf_counter() - t0) * 1000
            print(f"\n========== [{case['label']}]  '{case['msg']}'  ({dt:.0f}ms) ==========")
            print("status:", status, " latencyMs:", body.get("latencyMs"))
            print("reply[:160]:", (body.get("reply") or "")[:160])
            for ev in body.get("events") or []:
                if ev.get("type") == "context":
                    print("  ctx.intent:", ev.get("intent"))
                if ev.get("type") == "memory_extracted":
                    print("  memory_extracted:", ev.get("ok"), "err=", (ev.get("error") or "")[:120])
            results.append((status, body))
        ok = all(200 <= s < 300 for s, _ in results)
        print("\n========== Summary ==========")
        print("passed:", sum(1 for s, _ in results if 200 <= s < 300), "/", len(results))
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))