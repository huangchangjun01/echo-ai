"""调试:完整流程 — 注册 → 注入 session → 进 /chat → 发消息 → 等响应 → 截图。"""
import asyncio
import json
import time
import uuid
from datetime import datetime
import httpx
from playwright.async_api import async_playwright


async def main() -> None:
    uname = f"debug_{uuid.uuid4().hex[:8]}"
    async with httpx.AsyncClient(base_url="http://localhost:8080", timeout=15.0) as c:
        r = await c.post("/api/auth/register",
                         json={"username": uname, "password": "test123", "nickname": "调试"})
        user_id = int(r.json()["data"]["id"])
        r = await c.post("/api/auth/login",
                         json={"username": uname, "password": "test123"})
        data = r.json()["data"]
        sid = data["sessionId"]
        try:
            dt = datetime.fromisoformat(data["expireAt"].replace("Z", "+00:00"))
            exp_ms = int(dt.timestamp() * 1000)
        except Exception:
            exp_ms = int(time.time() * 1000) + 86400_000
        user_info = data["user"]
        print(f"registered user_id={user_id} sid={sid[:12]}")

    payload = json.dumps({
        "sessionId": sid,
        "expiresAt": exp_ms,
        "user": user_info,
    }, ensure_ascii=False)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 800}, locale="zh-CN")
        page = await ctx.new_page()
        page.on("console", lambda msg: print(f"[console:{msg.type}]", msg.text[:200]))
        page.on("request", lambda req: print(f"[req] {req.method} {req.url[:120]}"))
        page.on("response", lambda res: print(f"[res] {res.status} {res.url[:120]}") if "/chat" in res.url else None)

        await page.add_init_script(
            f"() => {{ localStorage.setItem('echo_auth_session', {json.dumps(payload)}); }}"
        )
        print("navigating to /chat...")
        await page.goto("http://localhost:5173/chat", wait_until="networkidle", timeout=20_000)
        await asyncio.sleep(1)
        await page.screenshot(path="E:/AIWorking/workspace/report/echo/_dbg_1_loaded.png", full_page=True)

        # 检查 localStorage
        ls = await page.evaluate("() => localStorage.getItem('echo_auth_session')")
        print(f"localStorage echo_auth_session: {ls[:200] if ls else None}")

        # 检查 input
        ta = page.locator(".chat-input .el-textarea__inner, .empty-input .el-textarea__inner").first
        print(f"textarea visible: {await ta.is_visible()}")
        await ta.click()
        await ta.fill("我是一名数据科学家, 喜欢 Python")
        await asyncio.sleep(0.2)

        # 看 send 按钮
        btn = page.locator(".send-btn, .empty-send-btn").first
        print(f"send btn: visible={await btn.is_visible()} disabled={await btn.is_disabled()}")
        await page.screenshot(path="E:/AIWorking/workspace/report/echo/_dbg_2_before_send.png", full_page=True)

        await btn.click(force=True)
        print("clicked send")
        await asyncio.sleep(2)
        await page.screenshot(path="E:/AIWorking/workspace/report/echo/_dbg_3_after_2s.png", full_page=True)

        # 检查 typing-caret
        for i in range(20):
            await asyncio.sleep(1)
            caret = await page.evaluate("() => !!document.querySelector('.typing-caret')")
            bubbles = await page.evaluate(
                "() => Array.from(document.querySelectorAll('.message-bubble--assistant')).length"
            )
            content = await page.evaluate(
                "() => {"
                "  const items = document.querySelectorAll('.message-wrapper--assistant');"
                "  if (!items.length) return '';"
                "  const last = items[items.length - 1];"
                "  const bubble = last.querySelector('.message-content, .message-bubble');"
                "  return ((bubble && (bubble.innerText || bubble.textContent)) || '').trim().slice(0, 100);"
                "}"
            )
            print(f"  t={i+1}s caret={caret} bubbles={bubbles} content='{content}'")
            if content and not caret:
                break

        await page.screenshot(path="E:/AIWorking/workspace/report/echo/_dbg_4_done.png", full_page=True)
        await browser.close()


asyncio.run(main())