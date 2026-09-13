"""memory_relations 迁移工具:把历史「最近 id」建立的因果边,用 BGE-M3 向量相似度重选 source_id。

迁移策略 — 保守 / 干跑优先:
1. 默认 ``--dry-run``: 只读扫描 ``memory_relations``,对比新旧 source_id,输出差异统计 + 样例。
2. ``--apply`` 才真正 ``UPDATE``,且分批提交(默认每批 200 行),避免长事务。
3. ``--limit N``: 只迁移前 N 行,方便抽样验证。
4. ``--where user_id=xxx``: 限定到特定用户,做精准迁移。

不删除任何行,不动 EchoMemory 向量。回滚靠 source_id 备份字段 + ``memory_relations_backup`` 表。

用法:
    python -m tools.migrate_memory_relations_to_vector --dry-run
    python -m tools.migrate_memory_relations_to_vector --apply --limit 100
    python -m tools.migrate_memory_relations_to_vector --apply --user-id some_user
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import uuid
from typing import Any

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("migrate_memory_relations")


async def _create_backup_table() -> None:
    """备份 source_id 到独立表,保证可回滚。"""
    from database import execute

    await execute(
        """
        CREATE TABLE IF NOT EXISTS memory_relations_backup (
            id BIGINT PRIMARY KEY,
            user_id VARCHAR(128) NOT NULL,
            role_id VARCHAR(128) NOT NULL DEFAULT 'default',
            source_id BIGINT NOT NULL,
            target_id BIGINT NOT NULL,
            relation VARCHAR(16) NOT NULL,
            backup_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_backup_user (user_id)
        )
        """
    )


async def _load_relations(user_id: str | None, limit: int | None) -> list[dict]:
    from database import fetch_all

    where_sql = ""
    params: tuple = ()
    if user_id:
        where_sql = "WHERE user_id=%s"
        params = (user_id,)
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    rows = await fetch_all(
        f"""
        SELECT id, user_id, role_id, source_id, target_id, relation
        FROM memory_relations
        {where_sql}
        ORDER BY id ASC
        {limit_sql}
        """,
        params,
    )
    return list(rows)


async def _backup_rows(rows: list[dict]) -> None:
    if not rows:
        return
    from database import execute

    # 6 列: id, user_id, role_id, source_id, target_id, relation
    row_placeholder = "(%s, %s, %s, %s, %s, %s)"
    values_sql = ", ".join([row_placeholder] * len(rows))
    params: list = []
    for r in rows:
        params.extend([r["id"], r["user_id"], r["role_id"], r["source_id"], r["target_id"], r["relation"]])
    await execute(
        f"""
        INSERT IGNORE INTO memory_relations_backup
            (id, user_id, role_id, source_id, target_id, relation)
        VALUES {values_sql}
        """,
        tuple(params),
    )


async def _recompute_one(row: dict) -> dict:
    """对单条边调用新向量逻辑,返回 {old, new, changed}。"""
    from database import fetch_one
    from memory.extractor import _find_similar_cause

    target_row = await fetch_one(
        "SELECT content, level FROM memories WHERE id=%s",
        (row["target_id"],),
    )
    if not target_row:
        return {"old": row["source_id"], "new": None, "changed": False, "skipped": "target_missing"}
    new_source = await _find_similar_cause(
        user_id=row["user_id"],
        role_id=row.get("role_id") or "default",
        fact_text=target_row.get("content") or "",
        fact_level=target_row.get("level") or "L1",
        target_id=row["target_id"],
        relation=row.get("relation") or "",
    )
    return {
        "old": row["source_id"],
        "new": new_source,
        "changed": new_source is not None and new_source != row["source_id"],
        "skipped": None,
    }


async def _apply_one(row_id: int, new_source: int) -> None:
    from database import execute

    await execute(
        "UPDATE memory_relations SET source_id=%s WHERE id=%s",
        (new_source, row_id),
    )


async def run(
    *,
    dry_run: bool,
    apply: bool,
    limit: int | None,
    user_id: str | None,
    batch_size: int,
) -> None:
    await _create_backup_table()
    rows = await _load_relations(user_id=user_id, limit=limit)
    logger.info("loaded %d memory_relations rows", len(rows))
    if not rows:
        return

    # 备份 — 即使 dry_run 也写备份,这样后续 apply 时能复用同一快照
    await _backup_rows(rows)
    logger.info("backed up to memory_relations_backup")

    total = len(rows)
    changed = 0
    unchanged = 0
    skipped = 0
    samples: list[dict] = []
    pending_updates: list[tuple[int, int]] = []

    for i, row in enumerate(rows, 1):
        try:
            result = await _recompute_one(row)
        except Exception as e:
            logger.warning("recompute failed row id=%s: %s", row["id"], e)
            skipped += 1
            continue
        if result["skipped"]:
            skipped += 1
            continue
        if result["changed"]:
            changed += 1
            pending_updates.append((row["id"], result["new"]))
            if len(samples) < 10:
                samples.append(
                    {
                        "relation_id": row["id"],
                        "target_id": row["target_id"],
                        "relation": row["relation"],
                        "old_source": result["old"],
                        "new_source": result["new"],
                    }
                )
        else:
            unchanged += 1
        if i % 100 == 0:
            logger.info(
                "scanned %d/%d changed=%d unchanged=%d skipped=%d",
                i, total, changed, unchanged, skipped,
            )

    logger.info(
        "scan done: total=%d changed=%d unchanged=%d skipped=%d",
        total, changed, unchanged, skipped,
    )
    logger.info("sample changes (first 10):")
    for s in samples:
        logger.info("  rel_id=%s target=%s rel=%s %s -> %s",
                    s["relation_id"], s["target_id"], s["relation"],
                    s["old_source"], s["new_source"])

    if dry_run or not apply or not pending_updates:
        logger.info(
            "[DRY-RUN] %d updates proposed. Re-run with --apply to commit.",
            len(pending_updates),
        )
        return

    # apply
    committed = 0
    for start in range(0, len(pending_updates), batch_size):
        batch = pending_updates[start:start + batch_size]
        for row_id, new_source in batch:
            await _apply_one(row_id, new_source)
            committed += 1
        logger.info("applied batch %d-%d / %d",
                    start + 1, min(start + batch_size, len(pending_updates)),
                    len(pending_updates))
    logger.info("[APPLY] committed %d updates", committed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="只读扫描,不写入(默认)")
    parser.add_argument("--apply", action="store_true",
                        help="真正 UPDATE source_id")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 行")
    parser.add_argument("--user-id", type=str, default=None,
                        help="限定到单个 user_id")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="apply 时每批提交行数")
    args = parser.parse_args()
    apply = bool(args.apply)
    dry_run = not apply
    asyncio.run(run(
        dry_run=dry_run,
        apply=apply,
        limit=args.limit,
        user_id=args.user_id,
        batch_size=args.batch_size,
    ))


if __name__ == "__main__":
    main()