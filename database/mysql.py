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
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any

from config.config import get_settings
from utils.request_context import log_exception, log_silent_failure, merge_extra

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
except Exception:
    aiomysql = None  # type: ignore
    _HAS_AIOMYSQL = False

import pymysql  # 同步回退

_pool: Any | None = None
_pool_lock = asyncio.Lock()


async def _init_async_pool() -> Any:
    """创建 aiomysql 连接池。

    `pool_recycle` / `connect_timeout` 是防"陈旧连接"的第一道防线：
    远端 MySQL（wait_timeout）、NAT、防火墙都会回收长时间空闲的 TCP 连接，
    而 aiomysql 的 `_fill_free_pool` 只能剔除 `at_eof()` / `eof_received` 的连接，
    半开连接会被原样发出去，让下一条 SQL 阻塞到 OS 超时才报 2013。
    """
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
        pool_recycle=cfg.pool_recycle,
        connect_timeout=cfg.connect_timeout,
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
        connect_timeout=cfg.connect_timeout,
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
    except Exception as e:
        log_exception(
            logger,
            "close pool failed",
            exc=e,
            level=logging.WARNING,
            stage="db_pool",
            event="close_error",
            backend="aiomysql" if _HAS_AIOMYSQL else "pymysql_threadpool",
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


# ---------- 连接丢失重试 ----------

# 这些 errno 都表示"连接本身没了"，而不是 SQL 有问题 —— 换一条连接重放即可。
# 2006 MySQL server has gone away / 2013 Lost connection during query
# 2055 Lost connection to server (async) / 4031 服务端主动踢掉空闲会话
_RETRYABLE_ERRNOS = frozenset({2006, 2013, 2055, 4031})

# 总尝试次数（含首次）。远端库一般只可能有少量陈旧连接，3 次足够；
# 再多只会在真·数据库宕机时放大故障时间。
_DB_MAX_ATTEMPTS = 3


def _is_connection_lost(exc: BaseException) -> bool:
    """判断异常是否为"连接已失效"，即可以安全地换连接重试。

    注意只认连接类错误：语法错误、约束冲突等重试多少次都一样，必须立刻抛出。
    """
    if isinstance(exc, pymysql.err.InterfaceError):
        return True
    if isinstance(exc, pymysql.err.OperationalError):
        code = exc.args[0] if exc.args else None
        return code in _RETRYABLE_ERRNOS
    return False


async def _purge_free_connections() -> None:
    """丢弃池中所有空闲连接。

    连接失效通常是"整段空闲期链路被回收"造成的，池里其余空闲连接大概率同样是
    半开状态。若不清掉，重试会再挑一条陈旧连接，又阻塞一个 OS 超时（线上实测 ~21s）。
    这里只做 transport.close()（非阻塞），不发 COM_QUIT——往死连接写数据同样会挂。
    """
    pool = _pool
    free = getattr(pool, "_free", None)
    if free is None:  # pymysql 回退路径每次新建连接，无需清理
        return
    while free:
        conn = free.pop()
        try:
            conn.close()
        except Exception as e:
            log_silent_failure(
                logger,
                "close stale connection failed",
                exc=e,
                stage="db",
                event="purge_error",
            )


async def _run_with_retry(
    op: str,
    sql: str,
    params: tuple | dict | None,
    runner: Callable[[], Coroutine[Any, Any, tuple[Any, int]]],
) -> Any:
    """执行 `runner()`（返回 `(结果, 行数)`），遇到连接失效时换连接重试。

    统一承担慢 SQL / 成功 / 失败日志，避免三个 API 各写一份。
    """
    sql_summary = _summary_sql(sql)
    param_count = _count_params(params)
    for attempt in range(1, _DB_MAX_ATTEMPTS + 1):
        t0 = time.perf_counter()
        try:
            result, rows = await runner()
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            if _is_connection_lost(e) and attempt < _DB_MAX_ATTEMPTS:
                log_exception(
                    logger,
                    "db connection lost, retrying with a fresh connection",
                    exc=e,
                    level=logging.WARNING,
                    stage="db",
                    event="retry",
                    op=op,
                    sql=sql_summary,
                    params=param_count,
                    attempt=attempt,
                    max_attempts=_DB_MAX_ATTEMPTS,
                    duration_ms=round(elapsed, 2),
                )
                await _purge_free_connections()
                continue
            log_exception(
                logger,
                f"db {op} failed",
                exc=e,
                level=logging.ERROR,
                stage="db",
                event="error",
                op=op,
                sql=sql_summary,
                params=param_count,
                attempt=attempt,
                duration_ms=round(elapsed, 2),
            )
            raise
        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.log(
            logging.WARNING if elapsed >= _SLOW_SQL_MS else logging.DEBUG,
            "slow sql" if elapsed >= _SLOW_SQL_MS else f"sql {op}",
            extra=merge_extra(
                stage="db",
                event="slow" if elapsed >= _SLOW_SQL_MS else "ok",
                op=op,
                sql=sql_summary,
                params=param_count,
                rows=rows,
                attempt=attempt,
                duration_ms=round(elapsed, 2),
            ),
        )
        return result


async def execute(sql: str, params: tuple | dict | None = None) -> int:
    """执行单条 DML，返回 affected rows。"""

    async def _run() -> tuple[int, int]:
        async with acquire() as conn:
            if _HAS_AIOMYSQL and hasattr(conn, "cursor"):
                async with conn.cursor() as cur:
                    await cur.execute(sql, params or ())
                    rowcount = cur.rowcount
            else:
                def _sync() -> int:
                    with conn.cursor() as cur:  # type: ignore[attr-defined]
                        cur.execute(sql, params or ())
                        return cur.rowcount

                rowcount = await asyncio.to_thread(_sync)
        return rowcount, rowcount

    return await _run_with_retry("execute", sql, params, _run)


async def fetch_one(sql: str, params: tuple | dict | None = None) -> dict | None:
    async def _run() -> tuple[dict | None, int]:
        async with acquire() as conn:
            if _HAS_AIOMYSQL and hasattr(conn, "cursor"):
                async with conn.cursor() as cur:
                    await cur.execute(sql, params or ())
                    row = await cur.fetchone()
                    if row is not None:
                        cols = [d[0] for d in cur.description]
                        row = dict(zip(cols, row))
            else:
                def _sync() -> dict | None:
                    with conn.cursor() as cur:  # type: ignore[attr-defined]
                        cur.execute(sql, params or ())
                        row = cur.fetchone()
                        if row is None:
                            return None
                        cols = [d[0] for d in cur.description]
                        return dict(zip(cols, row))

                row = await asyncio.to_thread(_sync)
        return row, 1 if row else 0

    return await _run_with_retry("fetch_one", sql, params, _run)


async def fetch_all(sql: str, params: tuple | dict | None = None) -> list[dict]:
    async def _run() -> tuple[list[dict], int]:
        async with acquire() as conn:
            if _HAS_AIOMYSQL and hasattr(conn, "cursor"):
                async with conn.cursor() as cur:
                    await cur.execute(sql, params or ())
                    rows = await cur.fetchall()
                    if not rows:
                        rows_dict: list[dict] = []
                    else:
                        cols = [d[0] for d in cur.description]
                        rows_dict = [dict(zip(cols, r)) for r in rows]
            else:
                def _sync() -> list[dict]:
                    with conn.cursor() as cur:  # type: ignore[attr-defined]
                        cur.execute(sql, params or ())
                        rows = cur.fetchall()
                        if not rows:
                            return []
                        cols = [d[0] for d in cur.description]
                        return [dict(zip(cols, r)) for r in rows]

                rows_dict = await asyncio.to_thread(_sync)
        return rows_dict, len(rows_dict)

    return await _run_with_retry("fetch_all", sql, params, _run)


async def init_schema() -> None:
    """建表（幂等）。"""
    from database.schema import ensure_schema

    t0 = time.perf_counter()
    try:
        async with acquire() as conn:
            if _HAS_AIOMYSQL and hasattr(conn, "cursor"):
                async with conn.cursor() as cur:
                    await ensure_schema(cur)
            else:
                def _run() -> None:
                    with conn.cursor() as cur:  # type: ignore[attr-defined]
                        ensure_schema(cur)

                await asyncio.to_thread(_run)
    except Exception as e:
        log_exception(
            logger,
            "init_schema failed",
            exc=e,
            level=logging.ERROR,
            stage="init_schema",
            event="error",
            db_name=get_settings().db.name,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        raise
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
            except Exception as e:
                log_silent_failure(
                    logger,
                    "close pooled connection failed",
                    exc=e,
                    stage="db_close",
                    event="conn_close_error",
                )
        self._opened.clear()