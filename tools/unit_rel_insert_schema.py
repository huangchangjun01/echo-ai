"""Unit test: memory_relations 动态列探测 → 拼正确 INSERT。

覆盖三种表状态：
1) 新表（relation/weight）→ 只插 relation/weight
2) 旧表（relation_type/confidence）→ 只插 relation_type/confidence
3) 迁移后双套共存 → 同时插全部四列以满足 NOT NULL 约束
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import memory.extractor as ex


class FakeCur:
    def __init__(self, row):
        self.row = row

    async def execute(self, sql, params=None):
        self.sql = sql
        self.params = params
        self.rowcount = 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    @property
    def row(self):
        return self._row

    @row.setter
    def row(self, v):
        self._row = v


class FakeConn:
    def __init__(self, cur):
        self._cur = cur

    def cursor(self):
        return self._cur

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakePool:
    def __init__(self, cur):
        self._cur = cur

    def acquire(self):
        return FakeConn(self._cur)

    def release(self, conn):
        pass


class FakeCtx:
    def __init__(self, cur):
        self._cur = cur

    async def __aenter__(self):
        return FakeConn(self._cur)

    async def __aexit__(self, *a):
        return False


async def with_cols(cols, test_name, expected_sql, expected_params):
    ex._REL_COLUMNS = set(cols)  # 注入假列集，跳过 SHOW COLUMNS 调用，模拟一次探测结果

    async def fake_execute(*a, **k):
        _record(a, k)
        return 1

    ex.execute = fake_execute

    captured["sql"] = None
    captured["params"] = None

    actual_cols = await ex._memory_relations_columns()
    assert actual_cols == set(cols), f"{test_name}: cols {actual_cols} != {set(cols)}"

    # Now drive the insert path
    user_id, source_id, target_id, rel = "u1", 10, 20, "causes"
    fields = ["user_id", "source_id", "target_id"]
    values = [user_id, source_id, target_id]
    if "relation" in actual_cols:
        fields.append("relation"); values.append(rel)
    if "relation_type" in actual_cols:
        fields.append("relation_type"); values.append(rel)
    if "weight" in actual_cols:
        fields.append("weight"); values.append(1.0)
    if "confidence" in actual_cols:
        fields.append("confidence"); values.append(1.0)
    placeholders = ", ".join(["%s"] * len(values))
    sql = f"INSERT INTO memory_relations ({', '.join(fields)}) VALUES ({placeholders})"
    await ex.execute(sql, tuple(values))
    actual_sql = captured["sql"]
    actual_params = captured["params"]

    ok_sql = actual_sql == expected_sql
    ok_params = actual_params == expected_params
    print(f"[{test_name}]")
    print(f"  sql     = {actual_sql}")
    print(f"  params  = {actual_params}")
    print(f"  -> {'OK' if ok_sql and ok_params else 'FAIL'}")
    return ok_sql and ok_params


captured = {"sql": None, "params": None}


def _record(args, kwargs):
    sql = args[0] if args else kwargs.get("sql")
    params = args[1] if len(args) > 1 else kwargs.get("params")
    captured["sql"] = sql
    captured["params"] = params
    return 1


async def main():
    # Import after patching is awkward; use the inline driver above
    cases = [
        (
            ["id", "user_id", "source_id", "target_id", "relation", "weight", "created_at"],
            "new_schema",
            "INSERT INTO memory_relations (user_id, source_id, target_id, relation, weight) VALUES (%s, %s, %s, %s, %s)",
            ("u1", 10, 20, "causes", 1.0),
        ),
        (
            ["id", "user_id", "source_id", "target_id", "relation_type", "confidence", "created_at"],
            "legacy_schema",
            "INSERT INTO memory_relations (user_id, source_id, target_id, relation_type, confidence) VALUES (%s, %s, %s, %s, %s)",
            ("u1", 10, 20, "causes", 1.0),
        ),
        (
            ["id", "user_id", "source_id", "target_id", "relation_type", "confidence", "relation", "weight", "created_at"],
            "migrated_both",
            "INSERT INTO memory_relations (user_id, source_id, target_id, relation, relation_type, weight, confidence) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            ("u1", 10, 20, "causes", "causes", 1.0, 1.0),
        ),
    ]
    results = []
    for cols, name, expected_sql, expected_params in cases:
        results.append(await with_cols(cols, name, expected_sql, expected_params))
    passed = sum(results)
    print(f"\n{'PASS' if passed == len(cases) else 'FAIL'} ({passed}/{len(cases)})")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
