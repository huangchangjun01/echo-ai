"""Unit test: 规则快速路径覆盖度（不依赖 LLM）。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.intent import Intent, _rule_based_intent


CASES = [
    # (msg, expected_intent, expected_source)
    ("我回来了", Intent.CHAT, "rule_greeting"),
    ("你好", Intent.CHAT, "rule_greeting"),
    ("想看猫", Intent.IMAGE_SEARCH, "rule_image"),
    ("给我看猫的照片", Intent.IMAGE_SEARCH, "rule_image"),
    ("帮我找一张山的图片", Intent.IMAGE_SEARCH, "rule_image"),
    ("找张猫的图片", Intent.IMAGE_SEARCH, "rule_image"),
    ("之前那张照片还在吗", Intent.IMAGE_SEARCH, "rule_image"),
    ("看看上次那张图", Intent.IMAGE_SEARCH, "rule_image"),
    ("翻出那张图", Intent.IMAGE_SEARCH, "rule_image"),
    ("找上次的合同", Intent.DOC_SEARCH, "rule_doc"),
    ("翻出那份合同", Intent.DOC_SEARCH, "rule_doc"),
    ("找上次的简历", Intent.DOC_SEARCH, "rule_doc"),
    ("之前那份文件", Intent.DOC_SEARCH, "rule_doc"),
    ("找一篇笔记", Intent.TEXT_SEARCH, "rule_text"),
    ("之前那篇文章", Intent.TEXT_SEARCH, "rule_text"),
    ("我之前说过什么", Intent.RECALL, "rule_recall"),
    ("我上次讲过我喜欢猫", Intent.RECALL, "rule_recall"),
    ("我叫什么", Intent.RECALL, "rule_recall"),
    # 复合句 → 让 LLM 分类（None）
    ("我回来了，今天天气不错", None, None),
    ("我之前跟他说过我会去的", Intent.RECALL, "rule_recall"),
    ("今天我们去爬山吧", None, None),  # 纯陈述，无「想看/找/回忆」
]


def main() -> int:
    failed = 0
    for msg, exp_intent, exp_source in CASES:
        r = _rule_based_intent(msg)
        got_intent = r.intent if r else None
        got_source = r.source if r else None
        ok = got_intent == exp_intent and got_source == exp_source
        mark = "OK" if ok else "FAIL"
        print(f"[{mark}] {msg!r:35s}  -> intent={got_intent} source={got_source}  (exp {exp_intent}/{exp_source})")
        if not ok:
            failed += 1
    print(f"\n{'PASS' if failed == 0 else 'FAIL'} ({len(CASES) - failed}/{len(CASES)})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
