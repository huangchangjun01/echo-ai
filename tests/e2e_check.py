"""端到端接口自测。

字段名严格对齐 `app/agent_runner.py` 中的 Pydantic alias：
- /chat         : {userId, sessionId?, message, stream?}
- /persona POST : {userId, persona}
- /memory/extract: {userId, sessionId?, userMsg, assistantMsg}
- /memory/retrieve: {userId, query, topK?}
- /tools/call    : {name, args}
- /embedding     : {texts, modality?}
"""
from __future__ import annotations

import json
import sys
import time
from urllib import request as urlreq

BASE = "http://127.0.0.1:8000"
USER_ID = "alice"
SESSION_ID = f"sess-e2e-{int(time.time())}"


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
    except Exception as e:
        return 0, f"ERROR: {e}"


def _check(name: str, status: int, body, expected_ok: bool = True):
    ok = status == 200 and (not expected_ok or (isinstance(body, dict) and body.get("ok", True)))
    mark = "OK" if ok else "FAIL"
    print(f"[{mark}] {name}  status={status}")
    if isinstance(body, dict):
        s = json.dumps(body, ensure_ascii=False)
        if len(s) > 600:
            s = s[:600] + "..."
        # Console may not be UTF-8 (e.g. cp936 on Windows); use errors='replace' to avoid crashing on emoji.
        try:
            print(f"        {s}")
        except UnicodeEncodeError:
            print(f"        {s.encode('utf-8', errors='replace').decode('utf-8', errors='replace')}")
    else:
        try:
            print(f"        {body}")
        except UnicodeEncodeError:
            print(f"        {str(body).encode('utf-8', errors='replace').decode('utf-8', errors='replace')}")
    return ok


def main() -> int:
    failures = 0

    # 1) /health
    s, b = _get("/health")
    if not _check("1/9 GET /health", s, b):
        failures += 1

    # 2) POST /persona
    s, b = _post("/persona", {
        "userId": USER_ID,
        "persona": "你叫小回，是一个温暖、有同理心的中文 AI 伙伴，喜欢倾听并简短回应。",
    })
    if not _check("2/9 POST /persona", s, b):
        failures += 1

    # 3) GET /persona
    s, b = _get(f"/persona?user_id={USER_ID}")
    if not _check("3/9 GET /persona", s, b):
        failures += 1

    # 4) POST /memory/extract
    s, b = _post("/memory/extract", {
        "userId": USER_ID,
        "sessionId": SESSION_ID,
        "userMsg": "我今天去西湖散步，看到断桥上有只猫，心情突然变好了",
        "assistantMsg": "西湖确实很美，能让你心情变好就太好了。",
    })
    if not _check("4/9 POST /memory/extract", s, b):
        failures += 1

    # 5) POST /memory/retrieve
    s, b = _post("/memory/retrieve", {
        "userId": USER_ID,
        "query": "西湖散步",
        "topK": 5,
    })
    if not _check("5/9 POST /memory/retrieve", s, b):
        failures += 1

    # 6) POST /tools/call analyze_emotion
    s, b = _post("/tools/call", {
        "name": "analyze_emotion",
        "args": {
            "text": "我今天特别开心，因为我终于和好朋友们一起去了西湖散步",
            "user_id": USER_ID,
        },
    })
    if not _check("6/9 POST /tools/call analyze_emotion", s, b):
        failures += 1

    # 7) POST /tools/call search_memory
    s, b = _post("/tools/call", {
        "name": "search_memory",
        "args": {
            "query": "西湖",
            "user_id": USER_ID,
            "top_k": 5,
        },
    })
    if not _check("7/9 POST /tools/call search_memory", s, b):
        failures += 1

    # 8) POST /embedding
    s, b = _post("/embedding", {
        "texts": ["你好世界", "西湖散步"],
        "modality": "text",
    })
    if not _check("8/9 POST /embedding text", s, b):
        failures += 1

    # 9) POST /chat (collect)
    s, b = _post("/chat", {
        "userId": USER_ID,
        "sessionId": SESSION_ID,
        "message": "我上次提到过哪里让你心情变好？",
        "stream": False,
    }, timeout=120)
    if not _check("9/9 POST /chat", s, b):
        failures += 1

    print("")
    print(f"===== SUMMARY: {9 - failures}/9 OK, {failures} FAIL =====")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())