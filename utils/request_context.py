"""请求级上下文 + 日志辅助。

通过 ``ContextVar`` 在异步调用栈中传递 ``user_id / session_id / request_id``
等公共字段，``TextFormatter`` 会自动把它们追加到日志行末尾，无需每个
``extra=`` 都重复传。

公开 API：
    bind_request(**kw)            设置上下文，返回 Token
    unbind_request(token)         还原上下文
    current_context()             取当前上下文 dict（拷贝）
    current_context_in_thread()   线程里读 to_thread_with_ctx 注入的副本
    merge_extra(**kw)             生成可作为 ``logger.info(extra=...)`` 的 dict
    log_event(logger, msg, ...)   一行写日志，自动带 stage/event
    log_stage(logger, name, ...)  阶段级 context manager，yield 可写 dict
    request_context_scope(**kw)   with 语句风格的 bind/unbind 包装
    to_thread_with_ctx(func, ...) 跨 asyncio.to_thread 携带上下文

设计取舍：
    - ``ContextVar`` 在 ``asyncio.to_thread`` 跨线程时不会自动继承，
      因此提供 ``to_thread_with_ctx`` 让调用方显式同步；它内部用
      一个线程本地的 ContextVar 镜像，避免污染主线程状态。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from contextvars import ContextVar, Token
from typing import Any, Callable

# ContextVar 默认空 dict：getter 拿不到时返回 {}，避免 KeyError
_request_ctx: ContextVar[dict[str, Any]] = ContextVar("echo_request_ctx", default={})


# ---------- 线程本地 ContextVar 镜像（用于 to_thread_with_ctx） ----------

_thread_local = threading.local()


def _thread_local_ctx() -> ContextVar[dict[str, Any]]:
    """每个工作线程持有一个独立的 ContextVar，避免跨线程串状态。"""
    var = getattr(_thread_local, "ctx_var", None)
    if var is None:
        var = ContextVar("echo_thread_ctx", default={})
        _thread_local.ctx_var = var
    return var


# ---------- 上下文绑定 / 读取 ----------

def bind_request(**kwargs: Any) -> Token:
    """把字段写入当前上下文并返回 Token；用 ``unbind_request(token)`` 还原。

    同一作用域内多次调用会**累加**到现有上下文中；Token 只用于恢复本次
    调用之前的状态。
    """
    current = dict(_request_ctx.get() or {})
    current.update({k: v for k, v in kwargs.items() if v not in (None, "")})
    return _request_ctx.set(current)


def unbind_request(token: Token) -> None:
    _request_ctx.reset(token)


def current_context() -> dict[str, Any]:
    """主线程 / 异步上下文读法：返回当前上下文的浅拷贝。"""
    return dict(_request_ctx.get() or {})


def current_context_in_thread() -> dict[str, Any]:
    """在线程中读取 ``to_thread_with_ctx`` 注入的上下文（同步函数内部用）。"""
    var = _thread_local_ctx()
    return dict(var.get() or {})


@contextlib.contextmanager
def request_context_scope(**kwargs: Any):
    """with 语句风格的 ``bind_request`` / ``unbind_request`` 包装。"""
    token = bind_request(**kwargs)
    try:
        yield
    finally:
        unbind_request(token)


# ---------- 日志辅助 ----------

def merge_extra(**kwargs: Any) -> dict[str, Any]:
    """构造可作为 ``logger.info(extra=...)`` 的 dict。

    ContextVar 字段 (``user_id / session_id / request_id``) 不进 ``extra``，
    由 formatter 自动注入；调用方只关心 stage / event / 业务字段。
    """
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        out[k] = v
    return out


def log_event(
    logger: logging.Logger,
    msg: str,
    *,
    level: int = logging.INFO,
    event: str | None = None,
    **fields: Any,
) -> None:
    """一行写日志：自动带上 ``stage=`` 注入 + 当前上下文（由 formatter 处理）。"""
    extra = merge_extra(event=event, **fields)
    logger.log(level, msg, extra=extra)


@contextlib.contextmanager
def log_stage(
    logger: logging.Logger,
    name: str,
    *,
    start_msg: str | None = None,
    level: int = logging.INFO,
    **start_fields: Any,
):
    """阶段级 context manager：进入时打印 start，退出时打印 end + 耗时。

    yield 一个 dict，调用方可在过程中 ``ctx_meta.update(field=val)`` 把数据
    附加到 ``end`` 日志上。
    """
    t0 = time.perf_counter()
    ctx_meta: dict[str, Any] = {}
    if start_msg is not None:
        logger.log(
            level,
            start_msg,
            extra=merge_extra(stage=name, event="start", **start_fields),
        )
    try:
        yield ctx_meta
    except Exception as e:
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.log(
            logging.ERROR,
            f"{name} failed: {e}",
            extra=merge_extra(
                stage=name,
                event="error",
                duration_ms=elapsed,
                error=str(e)[:300],
                **ctx_meta,
            ),
        )
        raise
    else:
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        logger.log(
            level,
            f"{name} ok",
            extra=merge_extra(
                stage=name,
                event="end",
                ok=True,
                duration_ms=elapsed,
                **ctx_meta,
            ),
        )


# ---------- 跨线程传播 ----------

async def to_thread_with_ctx(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``asyncio.to_thread`` 替代：把当前 ContextVar 复制到工作线程后再跑。

    直接用 ``asyncio.to_thread`` 时 ContextVar 不会自动继承，工作线程里
    ``current_context()`` 会拿到空 dict。改用本函数后，业务可在同步函数
    内部调 ``current_context_in_thread()`` 读到一份只读副本。
    """
    ctx = current_context()
    loop = asyncio.get_running_loop()

    def _runner() -> Any:
        var = _thread_local_ctx()
        token = var.set(dict(ctx))
        try:
            return func(*args, **kwargs)
        finally:
            try:
                var.reset(token)
            except Exception:
                pass

    return await loop.run_in_executor(None, _runner)


__all__ = [
    "bind_request",
    "unbind_request",
    "current_context",
    "current_context_in_thread",
    "merge_extra",
    "log_event",
    "log_stage",
    "request_context_scope",
    "to_thread_with_ctx",
]