"""调试脚本:打开 /chat 页面,截图 + dump DOM。"""
import asyncio
from playwright.async_api import async_playwright


async def main() -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 800}, locale="zh-CN")
        page = await ctx.new_page()
        page.on("console", lambda msg: print(f"[console:{msg.type}]", msg.text[:200]))
        await page.goto("http://localhost:5173/chat", wait_until="networkidle", timeout=20_000)
        await asyncio.sleep(2)
        await page.screenshot(path="E:/AIWorking/workspace/report/echo/_debug_chat.png", full_page=True)
        # 列出 textarea / input
        info = await page.evaluate(
            "() => {"
            "  const tas = Array.from(document.querySelectorAll('textarea, .el-textarea__inner, [contenteditable]'))"
            "    .map(el => ({tag: el.tagName, cls: el.className, vis: !!el.offsetParent}));"
            "  const btns = Array.from(document.querySelectorAll('button, .el-button'))"
            "    .map(el => ({tag: el.tagName, cls: el.className, txt: (el.innerText||'').slice(0,40)}));"
            "  const url = location.href;"
            "  const title = document.title;"
            "  return {ta: tas, btns, url, title};"
            "}"
        )
        print("INFO:", info)
        await browser.close()


asyncio.run(main())