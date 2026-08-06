"""parse_summary 也必须对清洗后的 md 抽取正确。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skills import parse_summary

# 模拟真实清洗后的 md（来自 e2e_clean_dirty 测试的输出）
md = """# 塞纳河黄昏

## 摘要
黄昏时分在巴黎塞纳河畔散步，水面倒映着灯光。路旁有小提琴手演奏德彪西。

## 元数据
- 时间：黄昏
- 地点：巴黎塞纳河畔

## 记忆细节
### 片段 1 [来源: paris.txt]
在巴黎塞纳河畔散步...

## 记忆主观描述
无"""

summary = parse_summary(md)
assert "黄昏时分" in summary, f"summary extraction wrong: {summary!r}"
assert "## 元数据" not in summary, "summary bled into next section"
assert "## 记忆细节" not in summary
print(f"✅ parse_summary ok: {summary!r}")


# 防御性：脏数据（带 think）也能清洗并抽取
dirty = "<think>long thinking</think>\n# 测试\n\n## 摘要\n真正的摘要在这里\n\n## 元数据\n..."
summary2 = parse_summary(dirty)
assert "<think>" not in summary2
assert "真正的摘要" in summary2
assert "long thinking" not in summary2
print(f"✅ parse_summary 抗脏数据 ok: {summary2!r}")
