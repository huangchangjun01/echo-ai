"""E2E: 把 echo-core DB 里已存在的脏 md 拉下来，过一遍 _validate_and_patch，
再校验结果是否合法 markdown。这是用户报告的真实场景。"""

import sys
import urllib.request
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skills import _validate_and_patch

# 拉真实脏数据
DIRTY_MD = urllib.request.urlopen(urllib.request.Request(
    "http://localhost:8080/api/memory/md-content",
    data=json.dumps({"userId": "9", "memoryId": "37e231e6b61f471b97922ba5d44273e8"}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)).read().decode("utf-8")

print(f"=== 脏 md（前 300 字符） ===\n{DIRTY_MD[:300]}\n...")
print(f"脏 md 总长度: {len(DIRTY_MD)}")
assert "<think>" in DIRTY_MD, "expected dirty md to have think block"

# 清洗（topic 用 LLM 自己推断的 "塞纳河黄昏"）
clean = _validate_and_patch(DIRTY_MD, topic="塞纳河黄昏", subjective_desc="", source_files=["paris.txt"])

print(f"\n=== 清洗后 md（前 600 字符） ===\n{clean[:600]}\n...")
print(f"清洗后 md 总长度: {len(clean)}")

# 校验
assert "<think>" not in clean, "think block leaked after cleaning"
assert "long reasoning" not in clean, "thinking text leaked"
assert clean.count("# 塞纳河黄昏") == 1, f"title duplicated: count={clean.count('# 塞纳河黄昏')}"
for sec in ["## 摘要", "## 元数据", "## 记忆细节", "## 记忆主观描述"]:
    assert sec in clean, f"missing {sec}"
# 五大节之后不应有 <think> 残留
idx_think = DIRTY_MD.find("<think>")
if idx_think >= 0:
    idx_close = DIRTY_MD.find("</think>", idx_think)
    pre_think = DIRTY_MD[:idx_think].strip()
    # pre_think 是 patcher 补的 "# 标题"，应被去除
    assert not clean.startswith(pre_think[:50]) or "# 塞纳河黄昏" in clean[:200], "patcher dedup failed"

print("\n✅ E2E clean PASS: 脏 md 已被清洗为合法 markdown")
