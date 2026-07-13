"""Smoke test: image_search 意图 + 图片资源事件透传（需服务在跑）。"""

from __future__ import annotations

import asyncio
import io
import json
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import httpx

URL = "http://127.0.0.1:8000/chat"
USER = "000001"
SESSION = "t-image"


async def one(label: str, msg: str) -> dict:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as c:
        r = await c.post(
            URL,
            json={"userId": USER, "sessionId": SESSION, "message": msg, "stream": False},
        )
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"_raw": r.text[:300]}
        print(f"\n=== [{label}] {msg!r}  status={r.status_code} ===")
        evs = body.get("events") or []
        intent, source = "", ""
        tool_names, resource_count = [], 0
        for ev in evs:
            et = ev.get("type")
            if et == "context":
                intent = ev.get("intent")
                source = ev.get("intent_source")
            elif et == "tool":
                tool_names.append(ev.get("name"))
            elif et == "resource":
                resource_count += 1
        print(f"  intent={intent} source={source} tools={tool_names} resources={resource_count}")
        for ev in evs:
            if ev.get("type") == "resource":
                print(f"  resource url={ev.get('url')}")
                print(f"           name={ev.get('name')} modality={ev.get('modality')} mime={ev.get('mime_type')} file_id={ev.get('file_id')}")
        print(f"  reply[:160]={body.get('reply','')[:160]}")
        return body


async def main() -> int:
    results = {}
    r1 = await one("rule_image_cat", "想看猫")
    results["rule_image"] = r1.get("events") and any(e.get("type") == "resource" for e in r1["events"])
    r2 = await one("rule_image_mountain", "给我看山的照片")
    results["rule_image_2"] = r2.get("events") and any(e.get("type") == "resource" for e in r2["events"])
    r3 = await one("llm_image_complex", "我之前上传过一张猫的照片，能再看看吗？")
    results["llm_image"] = r3.get("events") and any(e.get("type") == "resource" for e in r3["events"])
    r4 = await one("rule_greeting_fast", "我回来了")
    results["greeting"] = True  # 不报错即通过

    print(f"\n=== Summary ===")
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
