"""Unit test: cascade_chat 真流式续写行为（mock client，无需服务器）。

覆盖：
1) 正常续写：大模型不重复前缀 → delta 即为续写内容。
2) 大模型重写前缀：门控去重跳过重复段，final 无重复。
3) 续写流含 <think>：增量剥离，delta 不含 think。
4) 空前缀：纯大模型流式。
关键断言：final == prefix + 所有 delta 之和（前缀绝不丢弃）。
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm.cascade as cascade


class FakeClient:
    def __init__(self, prefix: str, chunks: list[str]):
        self._prefix = prefix
        self._chunks = chunks

    async def small_prefix(self, messages, *, max_tokens=None, temperature=None):
        return self._prefix

    async def stream(self, messages, *, temperature=None, max_tokens=None):
        for c in self._chunks:
            await asyncio.sleep(0)
            yield c


async def run_case(name, prefix, chunks, expect_final):
    cascade.get_llm_client = lambda: FakeClient(prefix, chunks)
    got_prefix = None
    deltas = []
    final = None
    async for ev in cascade.cascade_chat([{"role": "user", "content": "hi"}]):
        if ev["type"] == "prefix":
            got_prefix = ev["text"]
        elif ev["type"] == "delta":
            deltas.append(ev["text"])
        elif ev["type"] == "done":
            final = ev["full"]

    joined = (got_prefix or "") + "".join(deltas)
    ok_consistency = joined == final          # final 恒等于 prefix + 所有 delta
    ok_final = final == expect_final
    ok_prefix = got_prefix == cascade._strip_think(prefix)
    ok = ok_consistency and ok_final and ok_prefix
    print(f"[{name}] prefix={got_prefix!r} deltas={deltas} final={final!r}")
    print(f"    consistency={ok_consistency} final_match={ok_final} prefix_match={ok_prefix} -> {'OK' if ok else 'FAIL'}")
    return ok


async def main():
    cases = [
        ("normal_continuation", "你好，", ["今天", "天气", "不错"], "你好，今天天气不错"),
        ("big_rewrites_prefix", "你好，", ["你好", "，今天", "很好"], "你好，今天很好"),
        ("think_in_stream", "嗯，", ["<think>", "分析用户", "</think>", "我在", "听"], "嗯，我在听"),
        ("empty_prefix", "", ["直接", "流式", "输出"], "直接流式输出"),
    ]
    results = []
    for c in cases:
        results.append(await run_case(*c))
    passed = sum(results)
    print(f"\n{'PASS' if passed == len(cases) else 'FAIL'} ({passed}/{len(cases)})")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
