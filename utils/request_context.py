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
import sys
import threading
import time
from contextvars import ContextVar, Token
from typing import Any, Callable

logger = logging.getLogger(__name__)

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


_ERR_TEXT_LIMIT = 1000


def _format_exc_details(exc: BaseException) -> dict[str, Any]:
    """把异常对象的关键信息抽出来注入 extras，便于一行搜索定位。

    返回：
        error_type    - 异常类型名（用于 grep / 报警规则）
        error_msg     - 异常字符串（截到 1000 字符以免吞字段）
        error_module  - 抛出异常的模块路径
        error_class   - 异常类的全限定名（"module.QualName"）
    """
    cls = type(exc)
    module = getattr(cls, "__module__", "") or ""
    qualname = getattr(cls, "__qualname__", "") or cls.__name__
    err_text = str(exc) or repr(exc)
    if len(err_text) > _ERR_TEXT_LIMIT:
        err_text = err_text[: _ERR_TEXT_LIMIT - 3] + "..."
    return {
        "error_type": cls.__name__,
        "error_msg": err_text,
        "error_module": module,
        "error_class": f"{module}.{qualname}" if module else qualname,
    }


def log_exception(
    logger: logging.Logger,
    msg: str,
    *,
    exc: BaseException | None = None,
    level: int = logging.ERROR,
    include_traceback: bool = True,
    **fields: Any,
) -> None:
    """详细异常日志：自动注入 ``error_type`` / ``error_msg`` / ``error_module``
    并附带完整 traceback。

    用于替代散落的 ``logger.warning("xxx failed: %s", e)`` 写法，集中输出
    异常类型、消息、模块、堆栈与调用方提供的业务上下文（``stage`` / ``user_id`` /
    ``url`` / ``sql`` 等）。

    设计原则（按用户要求）：
    - **关键信息齐全**：除 traceback 外，还显式注入 ``error_type`` / ``error_msg``
      / ``error_module`` / ``error_class``，便于在日志聚合平台按 ``error_type``
      检索或报警；任何调用方传入的 ``**fields``（``url`` / ``sql`` / ``tool`` …
      ）也会原样保留。
    - **完整堆栈**：默认 ``include_traceback=True``；调用方如果不希望刷屏可显式
      传 ``include_traceback=False``，但绝大多数"出错需要排查"场景都应保留堆栈。
    - **活跃异常自动捕获**：没显式传 ``exc`` 时取 ``sys.exc_info()[1]``，因此
      在 ``except Exception as e:`` 里直接 ``log_exception(logger, "...")``
      也能拿到 traceback（依赖 ``logger.log(..., exc_info=...)`` 触发系统抓取）。

    参数：
        exc: 显式传入的异常；缺省时自动取 ``sys.exc_info()[1]``（必须是异常
            活跃状态下调用，否则取到 ``None``）。
        level: 日志级别，``logger.exception`` 仅 ERROR 级自带堆栈，其他级别
            需要显式 ``exc_info=True``。
        include_traceback: 是否附带 ``exc_info``；DEBUG 级以下通常关闭以免刷屏。
        fields: 业务字段，全部走 ``merge_extra``，``None`` / 空字符串会被剔除。
            调用方传入的字段名 **不要** 用 ``error_type`` / ``error_msg`` /
            ``error_module`` / ``error_class`` —— 这四个字段由本函数独占注入。
    """
    fields = dict(fields)
    if exc is not None:
        for k, v in _format_exc_details(exc).items():
            fields.setdefault(k, v)
    if include_traceback:
        logger.log(
            level,
            msg,
            exc_info=exc is not None or sys.exc_info()[0] is not None,
            extra=merge_extra(**fields),
        )
    else:
        logger.log(level, msg, extra=merge_extra(**fields))


def log_silent_failure(
    logger: logging.Logger,
    msg: str,
    *,
    exc: BaseException | None = None,
    level: int = logging.DEBUG,
    include_traceback: bool = False,
    **fields: Any,
) -> None:
    """用于"必须 swallow 但仍要可观测"的路径，DEBUG 级记录，不污染 INFO。

    适用场景：
    - 关闭游标 / 关闭连接 / 释放资源等 cleanup 路径
    - fallback 分支（chunk 切分 / 编码探测）失败不影响主流程
    - SSE 流式 chunk 解析、JSON 抽取等"跳过这一段"的循环内吞异常

    与 ``log_exception`` 的差异：不强制带 traceback，默认 DEBUG；想要 INFO
    级提示时显式传 ``level=logging.INFO``。如果排查需要完整堆栈（很罕见），
    可显式传 ``include_traceback=True``。
    """
    fields = dict(fields)
    if exc is not None:
        for k, v in _format_exc_details(exc).items():
            if k == "error_msg":
                # silent 路径里信息量小一些，error_msg 截到 200 字
                err_text = v
                if len(err_text) > 200:
                    err_text = err_text[:197] + "..."
                fields["error_msg"] = err_text
            else:
                fields.setdefault(k, v)
    if include_traceback:
        logger.log(
            level,
            msg,
            exc_info=exc is not None or sys.exc_info()[0] is not None,
            extra=merge_extra(**fields),
        )
    else:
        logger.log(level, msg, extra=merge_extra(**fields))


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
            except Exception as e:
                log_silent_failure(
                    logger,
                    "to_thread runner: thread context reset failed (token already consumed?)",
                    exc=e,
                    stage="to_thread_ctx",
                    event="reset_error",
                )

    return await loop.run_in_executor(None, _runner)


__all__ = [
    "bind_request",
    "unbind_request",
    "current_context",
    "current_context_in_thread",
    "merge_extra",
    "log_event",
    "log_exception",
    "log_silent_failure",
    "log_stage",
    "request_context_scope",
    "to_thread_with_ctx",
]