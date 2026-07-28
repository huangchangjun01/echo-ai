"""自测：验证记忆解析修复（fileKey 优先，产出 markdown 而非 HTML）。

覆盖本次改动：
- biz.recall._parse_one 透传真实 fileKey
- parsers.text_parser 优先用 fileKey 直连七牛
- utils.downloader.fetch_source_bytes fileKey 优先、url 兜底
- skills.build_memory_md 产出 markdown

真实依赖：七牛云（上传/下载源文件）+ MiniMax LLM（生成 md）。
不写数据库、不写向量、不删除任何数据。
"""
from __future__ import annotations

import asyncio
import io
import sys

# Windows GBK 控制台无法输出 emoji/部分中文，强制 UTF-8，避免自测末尾 UnicodeEncodeError
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from storage.qiniu_client import upload_bytes
from parsers.registry import parse_file
from biz.recall import _parse_one
from skills import build_memory_md

TEST_KEY = "memory/_selftest/parse_fix/source_note.txt"
REAL_CONTENT = (
    "2026年7月20日傍晚，在杭州西湖边，和老王、小李一起看了日落。\n"
    "我们聊了创业的事，心情很放松。湖面的荷花开得正好，还拍了合影。\n"
)
# 故意给一个错误的 url（模拟 SPA fallback / 过期签名会返回 HTML 页面）。
# 修复后 fileKey 优先，url 不应被使用。
BOGUS_URL = "http://the04ztre.hn-bkt.clouddn.com/this-object-does-not-exist-xxx"

HTML_MARKERS = ("<html", "<!doctype", "<body", "<div id=\"root\"", "<script")


def _has_html(s: str) -> bool:
    low = (s or "").lower()
    return any(m in low for m in HTML_MARKERS)


async def main() -> int:
    print("== step 1: 上传真实源文件到七牛 ==")
    upload_bytes(TEST_KEY, REAL_CONTENT.encode("utf-8"), mime_type="text/plain; charset=utf-8")
    print(f"   uploaded key={TEST_KEY} bytes={len(REAL_CONTENT.encode('utf-8'))}")

    print("== step 2: parse_file 直接用 fileKey（url 为错误地址）==")
    parsed = await parse_file(TEST_KEY, "source_note.txt", 1, BOGUS_URL)
    detail = parsed.detail_md or ""
    print(f"   modality={parsed.modality} detail_len={len(detail)} meta={parsed.meta}")
    print(f"   detail_head={detail[:80]!r}")
    assert "西湖" in detail, "FAIL: fileKey 下载的正文缺失（未走 fileKey 路径？）"
    assert not _has_html(detail), f"FAIL: 解析结果含 HTML: {detail[:120]!r}"
    print("   OK: fileKey 路径拿到真实正文，且无 HTML")

    print("== step 3: _parse_one 透传 fileKey（回归本次核心修复）==")
    name, p2 = await _parse_one({
        "fileKey": TEST_KEY,
        "fileName": "source_note.txt",
        "fileType": 1,
        "url": BOGUS_URL,
    })
    d2 = p2.detail_md or ""
    assert "西湖" in d2 and not _has_html(d2), f"FAIL: _parse_one 未透传 fileKey: {d2[:120]!r}"
    print("   OK: _parse_one 正确透传 fileKey")

    print("== step 4: build_memory_md 生成 markdown（真实 LLM）==")
    md = await build_memory_md(
        topic="2026-07-20 杭州西湖看日落",
        subjective_desc="和老王小李一起，很放松",
        details=[{"fileName": "source_note.txt", "detail": detail}],
        source_files=["source_note.txt"],
        use_large_model=False,
    )
    print("   ---- md head ----")
    print("\n".join(md.splitlines()[:12]))
    print("   ---- full md ----")
    print(md)
    print("   ---- end md ----")
    assert md.lstrip().startswith("#"), "FAIL: md 不是以标题开头"
    assert "## 摘要" in md, "FAIL: md 缺少 ## 摘要 结构"
    assert not _has_html(md), f"FAIL: 生成的 md 含 HTML: {md[:200]!r}"
    print(f"   OK: md 是 markdown（len={len(md)}），不含 HTML")

    print("\nALL PASS ✅ 记忆解析产出为 markdown，fileKey 优先修复生效")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
