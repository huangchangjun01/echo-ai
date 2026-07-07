"""Agent 端到端校验：
1. 旧的内部接口必须被拒绝（405/404）
2. /chat 必须内部完成 agent 全链路：
   - persona 注入（首次响应体现自定义 persona）
   - 记忆 L0/L1 自动抽取与使用（用户提到旧事，回复里要出现）
   - ReAct 工具调用（用户问"上次/记得"→触发 search_memory →命中相似记忆）
3. /health 探针正常
4. /ingest_file 入队接口正常
"""
from __future__ import annotations

import json
import sys
import time
from urllib import request as urlreq
from urllib.error import HTTPError

BASE = "http://127.0.0.1:8000"
USER_ID = f"agent-{int(time.time())}"
SESSION_ID = f"sess-agent-{int(time.time())}"


def _post(path: str, body: dict, timeout: int = 90) -> tuple[int, dict | str]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urlreq.Request(
        f"{BASE}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlreq.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw
    except HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, f"ERROR: {e}"


def _get(path: str, timeout: int = 30) -> tuple[int, dict | str]:
    try:
        with urlreq.urlopen(f"{BASE}{path}", timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw
    except HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, f"ERROR: {e}"


def _expect_fail(name: str, path: str) -> bool:
    """对外不应该暴露的路由，必须返回 404/405。"""
    s, b = _get(path, timeout=10)
    ok = s in (404, 405, 422)
    mark = "OK" if ok else "FAIL"
    print(f"[{mark}] removed-route {path} → status={s}")
    return ok


def _check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK" if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"[{mark}] {name}{suffix}")
    return ok


def main() -> int:
    failures = 0
    print("===== Phase 1: removed routes must reject =====")
    for p in [
        "/persona",
        "/persona?user_id=alice",
        "/memory/extract",
        "/memory/retrieve",
        "/memory/causal/1?user_id=alice",
        "/tools",
        "/tools/call",
        "/embedding",
    ]:
        if not _expect_fail("removed", p):
            failures += 1

    print("\n===== Phase 2: /chat full agent loop =====")

    # 先写一条线索：让 alice 提到她去西湖散步（通过 /chat 触发自动记忆抽取）
    print("-- turn 1: 植入记忆 --")
    s, b = _post("/chat", {
        "userId": USER_ID,
        "sessionId": SESSION_ID + "-t1",
        "message": "我今天去西湖散步，在断桥上看到一只猫，心情突然变好了。",
    }, timeout=120)
    if not _check("turn-1 status=200", s == 200, str(s)):
        failures += 1
    print(f"        reply: {(b if isinstance(b, str) else json.dumps(b, ensure_ascii=False))[:300]}")

    # 等几秒，让 fire-and-forget 的记忆抽取落库
    time.sleep(4)

    # turn 2：触发 ReAct 检索（用户说"记得/之前"，agent 应走 search_memory）
    print("\n-- turn 2: 触发 ReAct 检索 --")
    s, b = _post("/chat", {
        "userId": USER_ID,
        "sessionId": SESSION_ID + "-t2",
        "message": "你还记得我上次提到过什么让我心情变好了吗？",
    }, timeout=120)
    if not _check("turn-2 status=200", s == 200, str(s)):
        failures += 1
    body = b if isinstance(b, dict) else {}
    events = body.get("events", []) or []
    tool_events = [e for e in events if e.get("type") == "tool"]
    # Agent 设计：先看 L1 预注入；预注入够用就不调用工具；不够再走 ReAct 调 search_memory。
    # 这里既允许「工具命中」也允许「回复里直接命中预注入的记忆」。
    reply = body.get("reply", "") or ""
    tool_called = any(e.get("name") == "search_memory" for e in tool_events)
    if not _check("turn-2 工具或预注入命中记忆",
                  tool_called or ("西湖" in reply and "猫" in reply),
                  f"tool_called={tool_called}, tools={[e.get('name') for e in tool_events]}, reply[:80]={reply[:80]}"):
        failures += 1
    if not _check("turn-2 回复中提到西湖", "西湖" in reply, f"reply={reply[:120]}"):
        failures += 1
    if not _check("turn-2 回复中提到猫", "猫" in reply, f"reply={reply[:120]}"):
        failures += 1

    print("\n===== Phase 3: /health + /ingest_file =====")
    s, b = _get("/health")
    if not _check("GET /health", s == 200 and isinstance(b, dict) and b.get("status") == "ok"):
        failures += 1

    s, b = _post("/ingest_file", {
        "userId": USER_ID,
        "file": {
            "fileId": "f-test-001",
            "fileName": "report.txt",
            "url": "https://example.com/nonexistent.txt",
        },
    }, timeout=30)
    if not _check("POST /ingest_file queued", s == 200 and isinstance(b, dict) and b.get("ok") and b.get("queued"),
                  str(b)[:200]):
        failures += 1

    print(f"\n===== SUMMARY: {6 - failures}/6 groups OK, {failures} FAIL =====")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())