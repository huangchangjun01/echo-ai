"""Unit test for extractor.py's LLM-output parsing logic.

Synthesizes the two failure modes that previously broke JSON parsing:
1) Think block contains a `[` (e.g. enumerated reasoning notes)
2) Think block contains a `{` (e.g. pseudo JSON in reasoning)

Without _strip_think the greedy regex would have swallowed the think block.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.cascade import _strip_think

# Re-implement the same regex shape used in extractor for the test.
_RE_LIST = re.compile(r"\[[\s\S]*?\]")
_RE_OBJ = re.compile(r"\{[\s\S]*?\}")


CASES = [
    {
        "name": "think_with_inline_array_then_real_json",
        "raw": (
            "<think>The user mentioned they climbed a mountain. "
            "Let me list the facts: [climbed mountain, with friend].</think>\n"
            "[{\"fact\": \"用户和朋友去爬了山\", \"causes\": \"\", \"level\": \"L1\", "
            "\"emotion\": \"happy\", \"intensity\": 0.6, \"relation\": \"\"}]"
        ),
        "kind": "list",
    },
    {
        "name": "think_with_inline_object_then_real_json",
        "raw": (
            "<think>The two memories are similar because they both mention爬山, "
            "but different mountains. My answer: {\"duplicate\": false}.</think>\n"
            "{\"duplicate\": false, \"merged\": \"周末和朋友去爬了西山\", \"relation\": \"\"}"
        ),
        "kind": "obj",
    },
    {
        "name": "no_think_block_pure_json",
        "raw": '[{"fact": "user likes cats", "causes": "", "level": "L0"}]',
        "kind": "list",
    },
]


def main() -> int:
    import json as _json
    failed = 0
    for c in CASES:
        cleaned = _strip_think(c["raw"])
        regex = _RE_LIST if c["kind"] == "list" else _RE_OBJ
        m = regex.search(cleaned)
        ok = bool(m)
        try:
            parsed = _json.loads(m.group(0)) if m else None
        except Exception as e:
            parsed = None
            print(f"  parse_error: {e}")
        valid = ok and parsed is not None
        print(f"[{c['name']}] cleaned_len={len(cleaned)} matched={bool(m)} "
              f"parsed_type={type(parsed).__name__ if parsed is not None else 'None'} "
              f"-> {'OK' if valid else 'FAIL'}")
        if not valid:
            failed += 1
    print(f"\n{'PASS' if failed == 0 else 'FAIL'} ({len(CASES) - failed}/{len(CASES)})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())