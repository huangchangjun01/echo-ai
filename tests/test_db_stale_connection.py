"""MySQL 连接层的"陈旧连接"回归测试。

背景（线上故障）：
  /memory/parse 解析音频时，后台任务第一条 SQL
  `SELECT edit_status FROM recall_memory WHERE memory_id = %s`
  卡住 21.6 秒后抛
  `OperationalError(2013, 'Lost connection to MySQL server during query
   ([WinError 121] 信号灯超时时间已到)')`，导致整个解析链路直接失败。

根因：
  aiomysql 连接池长时间空闲后，池中的连接已被 MySQL server / NAT / 防火墙
  静默断开（半开 TCP）。aiomysql 的 `_fill_free_pool` 只能识别 `at_eof` /
  `eof_received`（即真正收到 FIN/RST）的连接；半开连接会被原样发出，读操作
  一直阻塞到 OS 超时（Windows 即 WinError 121）。
  而 `create_pool` 未设置 `pool_recycle`（默认 -1 = 永不回收），
  db 层也没有任何"连接丢失后重试"的兜底，于是首条 SQL 直接把后台任务打死。

本测试覆盖：
  1. 连接丢失类错误必须被识别为可重试；
  2. execute / fetch_one / fetch_all 遇到连接丢失时必须换连接重试并成功；
  3. 非连接类错误（如语法错误）不得重试，必须立即抛出；
  4. 连接池必须配置 pool_recycle，避免陈旧连接被发出来。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pymysql
import pytest

from database import mysql


class _FakeCursor:
    """最小可用的 aiomysql cursor 替身。"""

    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
        self.rowcount = 0
        self.description = (("edit_status",),)

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def execute(self, sql: str, params: object) -> None:
        self._conn.calls.append(sql)
        if self._conn.error is not None:
            raise self._conn.error
        self.rowcount = 1

    async def fetchone(self) -> tuple:
        return (0,)

    async def fetchall(self) -> list[tuple]:
        return [(0,)]


class _FakeConn:
    def __init__(self, error: Exception | None) -> None:
        self.error = error
        self.calls: list[str] = []

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)


def _lost_connection_error() -> pymysql.err.OperationalError:
    """复刻线上那条异常（errno 2013）。"""
    return pymysql.err.OperationalError(
        2013,
        "Lost connection to MySQL server during query ([WinError 121] 信号灯超时时间已到)",
    )


@pytest.fixture
def fake_pool(monkeypatch):
    """按脚本依次发出连接，并记录 acquire 次数。"""

    def _install(errors: list[Exception | None]) -> list[_FakeConn]:
        conns = [_FakeConn(e) for e in errors]
        handed_out: list[_FakeConn] = []

        @asynccontextmanager
        async def _fake_acquire():
            conn = conns[len(handed_out)]
            handed_out.append(conn)
            yield conn

        monkeypatch.setattr(mysql, "acquire", _fake_acquire)
        monkeypatch.setattr(mysql, "_HAS_AIOMYSQL", True)
        monkeypatch.setattr(mysql, "_purge_free_connections", _noop, raising=False)
        return handed_out

    async def _noop() -> None:
        return None

    return _install


async def test_lost_connection_is_retryable():
    assert mysql._is_connection_lost(_lost_connection_error()) is True
    assert mysql._is_connection_lost(pymysql.err.OperationalError(2006, "gone away")) is True
    assert mysql._is_connection_lost(pymysql.err.InterfaceError(0, "closed")) is True


async def test_syntax_error_is_not_retryable():
    assert mysql._is_connection_lost(pymysql.err.ProgrammingError(1064, "syntax")) is False


async def test_fetch_one_retries_after_stale_connection(fake_pool):
    """第一条连接是陈旧的，第二条正常 —— fetch_one 必须重试并返回结果。"""
    handed_out = fake_pool([_lost_connection_error(), None])

    row = await mysql.fetch_one(
        "SELECT edit_status FROM recall_memory WHERE memory_id = %s", ("m1",)
    )

    assert row == {"edit_status": 0}
    assert len(handed_out) == 2, "陈旧连接必须被丢弃并换一条新连接重试"


async def test_execute_retries_after_stale_connection(fake_pool):
    handed_out = fake_pool([_lost_connection_error(), None])

    affected = await mysql.execute(
        "UPDATE recall_memory SET edit_status = 1 WHERE memory_id = %s", ("m1",)
    )

    assert affected == 1
    assert len(handed_out) == 2


async def test_fetch_all_retries_after_stale_connection(fake_pool):
    handed_out = fake_pool([_lost_connection_error(), None])

    rows = await mysql.fetch_all("SELECT edit_status FROM recall_memory")

    assert rows == [{"edit_status": 0}]
    assert len(handed_out) == 2


async def test_non_connection_error_is_not_retried(fake_pool):
    handed_out = fake_pool([pymysql.err.ProgrammingError(1064, "syntax"), None])

    with pytest.raises(pymysql.err.ProgrammingError):
        await mysql.fetch_one("SELEC 1", ())

    assert len(handed_out) == 1, "非连接类错误不得重试"


async def test_retry_gives_up_when_all_connections_are_stale(fake_pool):
    """连续失败时不能无限重试，最终必须把原始异常抛出去。"""
    errors: list[Exception | None] = [_lost_connection_error() for _ in range(10)]
    handed_out = fake_pool(errors)

    with pytest.raises(pymysql.err.OperationalError):
        await mysql.fetch_one("SELECT 1", ())

    assert len(handed_out) == mysql._DB_MAX_ATTEMPTS


async def test_pool_is_created_with_recycle_and_connect_timeout(monkeypatch):
    """池必须配置 pool_recycle / connect_timeout，从源头避免发出陈旧连接。"""
    captured: dict = {}

    async def _fake_create_pool(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(mysql.aiomysql, "create_pool", _fake_create_pool)

    await mysql._init_async_pool()

    assert captured.get("pool_recycle", -1) > 0, "pool_recycle 未配置，陈旧连接会被发出来"
    assert captured.get("connect_timeout", 0) > 0, "connect_timeout 未配置，建连可能无限期挂起"
