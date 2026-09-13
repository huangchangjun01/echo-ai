"""调试:发送一条消息后,看看消息气泡的实际选择器。"""
import asyncio
from playwright.async_api import async_playwright


async def main() -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 800}, locale="zh-CN")
        page = await ctx.new_page()
        page.on("console", lambda msg: print(f"[console:{msg.type}]", msg.text[:200]))
        await page.goto("http://localhost:5173/chat", wait_until="networkidle", timeout=20_000)
        await asyncio.sleep(1)

        # 先发送一条
        ta = page.locator(".empty-input .el-textarea__inner").first
        await ta.click()
        await ta.fill("你好")
        btn = page.locator(".empty-send-btn").first
        await btn.click()

        # 等 3s 让消息流式渲染
        await asyncio.sleep(3)
        # 看 caret 和 bubble
        info = await page.evaluate(
            "() => {"
            "  const caret = document.querySelector('.typing-caret');"
            "  const bubbles = Array.from(document.querySelectorAll('[class*=\"bubble\"]'))"
            "    .map(el => ({cls: el.className, txt: (el.innerText||'').slice(0,80)}));"
            "  const msgs = Array.from(document.querySelectorAll('.message-item, .chat-message, [class*=\"message-item\"]'))"
            "    .map(el => ({cls: el.className}));"
            "  const allClasses = Array.from(new Set(Array.from(document.querySelectorAll('*'))"
            "    .map(el => el.className).filter(c => typeof c === 'string' && c.length < 60)))"
            "    .filter(c => /bubble|message|chat/i.test(c));"
            "  return {hasCaret: !!caret, bubbles, msgs, allClasses};"
            "}"
        )
        print("AFTER 3s:")
        print("  hasCaret:", info["hasCaret"])
        print("  bubbles:", info["bubbles"])
        print("  msgs:", info["msgs"])
        print("  classes:", info["allClasses"][:30])
        await page.screenshot(path="E:/AIWorking/workspace/report/echo/_debug_after_send.png", full_page=True)

        await browser.close()


asyncio.run(main())