"""端到端自测：注册/登录 → apply → 上传真实源文件到七牛 → save（触发解析）
→ 轮询 /detail 直到解析完成 → 断言 detail.mdContent 是 markdown 且不含 HTML。

覆盖三端改动：
- echo-ai：fileKey 优先解析 → 产出 markdown
- echo-core：/detail 新增下发 mdContent
- （前端渲染 mdContent 的逻辑等价于这里读取 detail.mdContent）
只创建一个测试用户 + 一条记忆，不删除任何数据。
"""
from __future__ import annotations

import sys
import time

import httpx
from qiniu import Auth, put_data  # type: ignore

CORE = "http://localhost:8080/api"
USER = "selftest_md_user"
PWD = "test123456"
ROLE = "default"

HTML_MARKERS = ("<!doctype", "<html", "@vite/client", "/src/main.ts", '<div id="app"')
SRC = (
    "2026年7月21日晚上，在上海外滩散步，看了黄浦江夜景和东方明珠灯光秀。\n"
    "和大学室友阿强重逢，聊了很多往事，非常开心。\n"
)


def has_html(s: str) -> bool:
    low = (s or "").lower()
    return any(m in low for m in HTML_MARKERS)


def main() -> int:
    c = httpx.Client(timeout=30)
    # 1) 注册（已存在则忽略）
    r = c.post(f"{CORE}/auth/register", json={"username": USER, "password": PWD, "nickname": "自测"})
    print(f"register -> {r.status_code} {r.json().get('message')}")
    # 2) 登录
    r = c.post(f"{CORE}/auth/login", json={"username": USER, "password": PWD})
    assert r.status_code == 200, f"login failed: {r.text}"
    sid = r.json()["data"]["sessionId"]
    H = {"X-Session-Id": sid}
    print(f"login ok sid={sid[:8]}...")

    # 3) apply memoryId
    r = c.post(f"{CORE}/memory/apply", headers=H, json={"sessionId": sid})
    mem = r.json()["data"]["memoryId"]
    print(f"apply memoryId={mem}")

    # 4) upload-token + 上传真实源文件到七牛
    fname = "waitan_note.txt"
    r = c.post(f"{CORE}/memory/upload-token", headers=H,
               json={"roleId": ROLE, "memoryId": mem, "fileName": fname, "isMd": False, "sessionId": sid})
    tok = r.json()["data"]
    key = tok["key"]
    print(f"upload-token key={key}")
    ret, info = put_data(tok["token"], key, SRC.encode("utf-8"), mime_type="text/plain")
    assert info.status_code == 200, f"qiniu upload failed: {info}"
    print(f"qiniu upload ok status={info.status_code}")

    # 5) save（触发 echo-ai 解析）
    topic = f"上海外滩夜景自测-{int(time.time())}"
    r = c.post(f"{CORE}/memory/save", headers=H, json={
        "memoryId": mem, "roleId": ROLE, "topic": topic, "subjectiveDesc": "和阿强重逢，很开心",
        "sourceFiles": [{"fileKey": key, "fileName": fname, "fileType": 1}], "sessionId": sid,
    })
    assert r.status_code == 200, f"save failed: {r.text}"
    print(f"save ok -> parseStatus={r.json()['data'].get('parseStatus')}")

    # 6) 轮询 /detail 直到解析完成（最多 90s）
    md = ""
    status = -1
    for i in range(45):
        time.sleep(2)
        r = c.get(f"{CORE}/memory/detail", headers=H, params={"memoryId": mem})
        d = r.json()["data"]
        status = d.get("parseStatus")
        md = d.get("mdContent") or ""
        print(f"  poll#{i} parseStatus={status} mdContent_len={len(md)} mdUrl={'Y' if d.get('mdUrl') else 'N'}")
        if status in (2, 3):
            break

    print("\n==== detail.mdContent 前 400 字 ====")
    print(md[:400])
    print("==== end ====")

    assert status == 2, f"FAIL: 解析未成功 parseStatus={status}"
    assert md, "FAIL: /detail 未返回 mdContent（后端字段未下发？）"
    assert not has_html(md), f"FAIL: mdContent 含 HTML: {md[:150]!r}"
    assert md.lstrip().startswith("#") and "##" in md, "FAIL: mdContent 不是 markdown 结构"
    print("\nE2E PASS: /detail.mdContent 为 markdown，无 HTML；三端链路修复生效")
    return 0


if __name__ == "__main__":
    sys.exit(main())
