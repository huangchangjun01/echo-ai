"""回归测试：build_memory_md 必须产出干净、合法的 markdown。

历史 bug：推理模型（如 MiniMax-M3）会把 chain-of-thought 写在 <think>...</think>
里输出，污染 md 文档；_validate_and_patch 还会重复补 `# {topic}` 标题。
"""

import asyncio
import sys
from pathlib import Path

# 让 tests/ 能 import 上层包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skills import _dedup_leading_title, _strip_think_and_fences, _validate_and_patch


# ===== _strip_think_and_fences =====

def test_strip_think_block_simple():
    md = "<think>thinking</think>\n# 标题\n## 摘要\nxx"
    out = _strip_think_and_fences(md)
    assert "<think>" not in out
    assert out.startswith("# 标题"), f"expected H1 at start, got: {out!r}"


def test_strip_think_block_multiline():
    md = "<think>\nline 1\nline 2\nline 3\n</think>\n# 标题"
    out = _strip_think_and_fences(md)
    assert "line 1" not in out
    assert "line 2" not in out
    assert out.lstrip().startswith("# 标题")


def test_strip_code_fence_markdown():
    md = "```markdown\n# 标题\n## 摘要\nxx\n```"
    out = _strip_think_and_fences(md)
    assert "```" not in out
    assert out.startswith("# 标题")


def test_strip_code_fence_md():
    md = "```md\n# 标题\n## 摘要\nxx\n```\n"
    out = _strip_think_and_fences(md)
    assert "```" not in out
    assert out.startswith("# 标题")


def test_strip_both_think_and_fence():
    md = "<think>thinking</think>\n```markdown\n# 标题\n## 摘要\nxx\n```"
    out = _strip_think_and_fences(md)
    assert "<think>" not in out
    assert "```" not in out
    assert out.startswith("# 标题")


def test_no_think_no_fence_passthrough():
    md = "# 标题\n## 摘要\n正常"
    out = _strip_think_and_fences(md)
    assert out == md


# ===== _dedup_leading_title =====

def test_dedup_when_topic_matches():
    md = "# 测试记忆\n## 摘要\nxx"
    out = _dedup_leading_title(md, "测试记忆")
    assert not out.startswith("# 测试记忆"), f"duplicate title not removed: {out!r}"
    assert out.startswith("## 摘要")


def test_no_dedup_when_topic_differs():
    md = "# 其他标题\n## 摘要\nxx"
    out = _dedup_leading_title(md, "测试记忆")
    assert out.startswith("# 其他标题")


def test_dedup_only_strips_exact_topic():
    md = "## 摘要\n# 测试记忆\n## 元数据\nxx"
    out = _dedup_leading_title(md, "测试记忆")
    # 不在第一行时不动
    assert out.startswith("## 摘要")


# ===== _validate_and_patch 端到端 =====

def test_validate_and_patch_strips_think():
    """真实 bug 场景：LLM 输出含 think 块 + patcher 重复标题"""
    llm_out = "<think>long reasoning here</think>\n# 测试记忆\n## 摘要\nxx\n## 元数据\n- 时间: 未知\n## 记忆细节\nxx\n## 记忆主观描述\nyy"
    out = _validate_and_patch(llm_out, topic="测试记忆", subjective_desc="yy", source_files=["a.jpg"])
    assert "<think>" not in out, "think block leaked"
    assert "long reasoning" not in out
    # 标题恰好出现一次（patcher 看到 LLM 已给 `# 测试记忆`，不应再补）
    assert out.count("# 测试记忆") == 1, f"title duplicated: {out!r}"
    # 五大节齐全
    for sec in ["## 摘要", "## 元数据", "## 记忆细节", "## 记忆主观描述"]:
        assert sec in out, f"missing section {sec}"


def test_validate_and_patch_strips_fence():
    llm_out = "```markdown\n# 测试记忆\n## 摘要\nxx\n```"
    out = _validate_and_patch(llm_out, topic="测试记忆", subjective_desc="", source_files=[])
    assert "```" not in out
    assert out.count("# 测试记忆") == 1
    assert "## 摘要" in out


def test_validate_and_patch_adds_missing_sections():
    """LLM 完全没有结构时，patcher 必须补齐全部五大节"""
    llm_out = "随便写点东西，没有任何标题"
    out = _validate_and_patch(llm_out, topic="测试", subjective_desc="主观", source_files=["f.png"])
    for sec in ["# 测试", "## 摘要", "## 元数据", "## 记忆细节", "## 记忆主观描述"]:
        assert sec in out, f"missing {sec}"
    # 来源应包含文件名
    assert "f.png" in out


def test_validate_and_patch_handles_empty_input():
    out = _validate_and_patch("", topic="空测试", subjective_desc="", source_files=[])
    assert out.startswith("# 空测试")
    for sec in ["## 摘要", "## 元数据", "## 记忆细节", "## 记忆主观描述"]:
        assert sec in out


if __name__ == "__main__":
    # 允许直接 `python tests/test_markdown_clean.py`
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
    sys.exit(1 if failed else 0)