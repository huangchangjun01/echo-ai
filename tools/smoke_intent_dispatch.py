"""Smoke test: 4 intent-driven dispatch cases.

- 我回来了                  → chat (rule_greeting), 无 tool hint, 不应调工具
- 帮我找张猫的图片            → image_search, 应注入 image hint
- 我之前说过什么            → recall, 应注入 search_memory hint
- 找上次的合同              → doc_search, 应注入 search_memory hint
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time

# Force UTF-8 stdout so Chinese / emoji replies don't crash Windows GBK console.
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import httpx  # noqa: E402

URL = "http://127.0.0.1:8000/chat"
USER = "intent-test-user"
SESSION = "smoke-session"

CASES = [
    {"label": "rule_greeting_chat",      "msg": "我回来了"},
    {"label": "llm_image_search",        "msg": "帮我找一张猫的图片"},
    {"label": "llm_recall",              "msg": "我之前说过什么？"},
    {"label": "llm_doc_search",          "msg": "找上次的合同"},
]


async def one(client: httpx.AsyncClient, msg: str) -> dict:
    body = {"userId": USER, "sessionId": SESSION, "message": msg, "stream": False}
    r = await client.post(URL, json=body, timeout=httpx.Timeout(60.0, connect=10.0))
    return {"status": r.status_code, "body": r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text[:300]}


async def main() -> int:
    async with httpx.AsyncClient() as c:
        out = []
        for case in CASES:
            t0 = time.perf_counter()
            res = await one(c, case["msg"])
            dt = (time.perf_counter() - t0) * 1000
            label = case["label"]
            print(f"\n========== [{label}]  '{case['msg']}'  ({dt:.0f}ms) ==========")
            print("status:", res["status"])
            body = res["body"]
            if not isinstance(body, dict):
                print("raw:", body)
                continue
            print("latencyMs:", body.get("latencyMs"))
            print("reply[:200]:", (body.get("reply") or "")[:200])
            events = body.get("events") or []
            for ev in events:
                et = ev.get("type")
                if et == "context":
                    print("  ctx.intent:", ev.get("intent"), "src=", ev.get("intent_source"),
                          "intent_ms=", ev.get("intent_ms"))
                elif et == "tool":
                    print("  tool:", ev.get("name"), "iter=", ev.get("iter"), "ok=", ev.get("ok"),
                          "sum=", (ev.get("summary") or "")[:120])
            out.append({"label": label, "ok": 200 <= res["status"] < 300, "events": events})
        failed = [c for c, r in zip(CASES, out) if not r["ok"]]
        print("\n========== Summary ==========")
        print("passed:", sum(1 for r in out if r["ok"]), "/", len(out))
        if failed:
            print("FAILED:", [f["label"] for f in failed])
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
