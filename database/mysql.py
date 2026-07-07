"""异步 MySQL 连接池（基于 aiomysql）。

- 进程内单例：第一次 `await get_pool()` 初始化，后续直接返回缓存的 pool。
- 提供同步助手 `run_sync`：把 sync DB 调用丢到默认 executor，避免阻塞事件循环。
- 不强依赖 aiomysql：若运行环境中未安装，自动回退到 pymysql 同步连接，但仍暴露
  协程风格的 API（在线程中执行），保证上层调用代码不变。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from config.config import get_settings
from utils.request_context import merge_extra

logger = logging.getLogger(__name__)

_SLOW_SQL_MS = 200.0  # 超过该耗时打 WARNING
_PARAM_NUM = re.compile(r"%s")


def _summary_sql(sql: str) -> str:
    """把 SQL 截短成 ~200 字符并隐藏参数，方便日志查看。"""
    s = re.sub(r"\s+", " ", sql or "").strip()
    return s[:200] + ("..." if len(s) > 200 else "")


def _count_params(params: Any) -> int:
    if params is None:
        return 0
    if isinstance(params, (list, tuple)):
        return len(params)
    if isinstance(params, dict):
        return len(params)
    return 1

try:  # 异步优先
    import aiomysql  # type: ignore

    _HAS_AIOMYSQL = True
except Exception:  # noqa: BLE001
    aiomysql = None  # type: ignore
    _HAS_AIOMYSQL = False

import pymysql  # 同步回退


_pool: Any | None = None
_pool_lock = asyncio.Lock()


async def _init_async_pool() -> Any:
    """创建 aiomysql 连接池。"""
    cfg = get_settings().db
    pool = await aiomysql.create_pool(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        db=cfg.name,
        minsize=cfg.pool_min,
        maxsize=cfg.pool_max,
        autocommit=True,
        charset="utf8mb4",
    )
    logger.info("aiomysql pool created: %s:%s/%s", cfg.host, cfg.port, cfg.name)
    return pool


def _sync_connect() -> Any:
    cfg = get_settings().db
    return pymysql.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        database=cfg.name,
        autocommit=True,
        charset="utf8mb4",
    )


async def get_pool() -> Any:
    """获取进程内单例连接池（异步优先；不可用时退到 pymysql 连接 + 线程池）。"""
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is not None:
            return _pool
        t0 = time.perf_counter()
        if _HAS_AIOMYSQL:
            _pool = await _init_async_pool()
        else:
            logger.warning(
                "aiomysql unavailable; falling back to pymysql (threadpool).",
                extra=merge_extra(stage="db_pool", event="fallback"),
            )
            _pool = _SyncPool()
        logger.info(
            "mysql pool ready",
            extra=merge_extra(
                stage="db_pool",
                event="ok",
                backend="aiomysql" if _HAS_AIOMYSQL else "pymysql_threadpool",
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is None:
        return
    t0 = time.perf_counter()
    try:
        if _HAS_AIOMYSQL and hasattr(_pool, "close"):
            _pool.close()
            await _pool.wait_closed()
        else:
            await asyncio.to_thread(_pool.close)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "close pool failed",
            extra=merge_extra(stage="db_pool", event="close_error", error=str(e)[:200]),
        )
    finally:
        _pool = None
        logger.info(
            "mysql pool closed",
            extra=merge_extra(
                stage="db_pool",
                event="closed",
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )


@asynccontextmanager
async def acquire() -> AsyncIterator[Any]:
    """`async with acquire() as conn:` 取一条连接。"""
    pool = await get_pool()
    if _HAS_AIOMYSQL and hasattr(pool, "acquire"):
        conn = await pool.acquire()
        try:
            yield conn
        finally:
            pool.release(conn)
    else:
        conn = await asyncio.to_thread(pool.get_connection)
        try:
            yield conn
        finally:
            await asyncio.to_thread(conn.close)


async def execute(sql: str, params: tuple | dict | None = None) -> int:
    """执行单条 DML，返回 affected rows。"""
    sql_summary = _summary_sql(sql)
    param_count = _count_params(params)
    t0 = time.perf_counter()
    try:
        async with acquire() as conn:
            if _HAS_AIOMYSQL and hasattr(conn, "cursor"):
                async with conn.cursor() as cur:
                    await cur.execute(sql, params or ())
                    rowcount = cur.rowcount
            else:
                def _run() -> int:
                    with conn.cursor() as cur:  # type: ignore[attr-defined]
                        cur.execute(sql, params or ())
                        return cur.rowcount

                rowcount = await asyncio.to_thread(_run)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if elapsed >= _SLOW_SQL_MS:
            logger.warning(
                "slow sql",
                extra=merge_extra(
                    stage="db",
                    event="slow",
                    op="execute",
                    sql=sql_summary,
                    params=param_count,
                    rowcount=rowcount,
                    duration_ms=round(elapsed, 2),
                ),
            )
        else:
            logger.debug(
                "sql executed",
                extra=merge_extra(
                    stage="db",
                    event="ok",
                    op="execute",
                    sql=sql_summary,
                    params=param_count,
                    rowcount=rowcount,
                    duration_ms=round(elapsed, 2),
                ),
            )
        return rowcount
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.exception(
            "db execute failed",
            extra=merge_extra(
                stage="db",
                event="error",
                op="execute",
                sql=sql_summary,
                params=param_count,
                duration_ms=round(elapsed, 2),
                error=str(e)[:300],
            ),
        )
        raise


async def fetch_one(sql: str, params: tuple | dict | None = None) -> dict | None:
    sql_summary = _summary_sql(sql)
    param_count = _count_params(params)
    t0 = time.perf_counter()
    try:
        async with acquire() as conn:
            if _HAS_AIOMYSQL and hasattr(conn, "cursor"):
                async with conn.cursor() as cur:
                    await cur.execute(sql, params or ())
                    row = await cur.fetchone()
                    if row is None:
                        cols = []
                    else:
                        cols = [d[0] for d in cur.description]
                        row = dict(zip(cols, row))
            else:
                def _run() -> dict | None:
                    with conn.cursor() as cur:  # type: ignore[attr-defined]
                        cur.execute(sql, params or ())
                        row = cur.fetchone()
                        if row is None:
                            return None
                        cols = [d[0] for d in cur.description]
                        return dict(zip(cols, row))

                row = await asyncio.to_thread(_run)
                cols = list(row.keys()) if isinstance(row, dict) else []
        elapsed = (time.perf_counter() - t0) * 1000.0
        if elapsed >= _SLOW_SQL_MS:
            logger.warning(
                "slow sql",
                extra=merge_extra(
                    stage="db",
                    event="slow",
                    op="fetch_one",
                    sql=sql_summary,
                    params=param_count,
                    rows=1 if row else 0,
                    duration_ms=round(elapsed, 2),
                ),
            )
        else:
            logger.debug(
                "sql fetch_one",
                extra=merge_extra(
                    stage="db",
                    event="ok",
                    op="fetch_one",
                    sql=sql_summary,
                    params=param_count,
                    rows=1 if row else 0,
                    duration_ms=round(elapsed, 2),
                ),
            )
        return row
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.exception(
            "db fetch_one failed",
            extra=merge_extra(
                stage="db",
                event="error",
                op="fetch_one",
                sql=sql_summary,
                params=param_count,
                duration_ms=round(elapsed, 2),
                error=str(e)[:300],
            ),
        )
        raise


async def fetch_all(sql: str, params: tuple | dict | None = None) -> list[dict]:
    sql_summary = _summary_sql(sql)
    param_count = _count_params(params)
    t0 = time.perf_counter()
    try:
        async with acquire() as conn:
            if _HAS_AIOMYSQL and hasattr(conn, "cursor"):
                async with conn.cursor() as cur:
                    await cur.execute(sql, params or ())
                    rows = await cur.fetchall()
                    if not rows:
                        cols = []
                        rows_dict: list[dict] = []
                    else:
                        cols = [d[0] for d in cur.description]
                        rows_dict = [dict(zip(cols, r)) for r in rows]
            else:
                def _run() -> list[dict]:
                    with conn.cursor() as cur:  # type: ignore[attr-defined]
                        cur.execute(sql, params or ())
                        rows = cur.fetchall()
                        if not rows:
                            return []
                        cols = [d[0] for d in cur.description]
                        return [dict(zip(cols, r)) for r in rows]

                rows_dict = await asyncio.to_thread(_run)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if elapsed >= _SLOW_SQL_MS:
            logger.warning(
                "slow sql",
                extra=merge_extra(
                    stage="db",
                    event="slow",
                    op="fetch_all",
                    sql=sql_summary,
                    params=param_count,
                    rows=len(rows_dict),
                    duration_ms=round(elapsed, 2),
                ),
            )
        else:
            logger.debug(
                "sql fetch_all",
                extra=merge_extra(
                    stage="db",
                    event="ok",
                    op="fetch_all",
                    sql=sql_summary,
                    params=param_count,
                    rows=len(rows_dict),
                    duration_ms=round(elapsed, 2),
                ),
            )
        return rows_dict
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.exception(
            "db fetch_all failed",
            extra=merge_extra(
                stage="db",
                event="error",
                op="fetch_all",
                sql=sql_summary,
                params=param_count,
                duration_ms=round(elapsed, 2),
                error=str(e)[:300],
            ),
        )
        raise


async def init_schema() -> None:
    """建表（幂等）。"""
    from database.schema import ensure_schema

    t0 = time.perf_counter()
    async with acquire() as conn:
        if _HAS_AIOMYSQL and hasattr(conn, "cursor"):
            async with conn.cursor() as cur:
                await ensure_schema(cur)
        else:
            def _run() -> None:
                with conn.cursor() as cur:  # type: ignore[attr-defined]
                    ensure_schema(cur)

            await asyncio.to_thread(_run)
    logger.info(
        "MySQL schema ensured",
        extra=merge_extra(
            stage="init_schema",
            event="ok",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        ),
    )


# ---------- pymysql 线程池回退包装 ----------

class _SyncPool:
    """当 aiomysql 不可用时，用 pymysql + asyncio.to_thread 模拟连接池。"""

    def __init__(self) -> None:
        self._sem = asyncio.Lock()
        self._opened: list[Any] = []
        self._closed = False

    def get_connection(self) -> Any:
        if self._closed:
            raise RuntimeError("MySQL pool closed")
        conn = _sync_connect()
        self._opened.append(conn)
        return conn

    def close(self) -> None:
        self._closed = True
        for c in self._opened:
            try:
                c.close()
            except Exception:
                pass
        self._opened.clear()