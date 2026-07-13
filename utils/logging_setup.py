"""日志设施：单行人类可读格式 + 可选文件输出。

设计上与 ``utils.request_context`` 协同：``TextFormatter`` 自动从
``ContextVar`` 取出 ``user_id / session_id / request_id`` 等字段，写到日志
行末尾，避免调用方在每个 ``extra=`` 里重复传。

使用约定：
    setup_logging()                # 应用启动时调用一次
    logger = get_logger(__name__)  # 业务模块入口
    logger.info("msg", extra=merge_extra(stage="...", event="...", ...))

环境变量：
    APP_LOG_LEVEL   默认 INFO
    LOG_TO_FILE     默认 true（写 logs/server.log）
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_CONFIGURED = False


def is_configured() -> bool:
    return _CONFIGURED


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class TextFormatter(logging.Formatter):
    """单行人类可读格式：``ts LEVEL logger [stage.event] [k=v ...] msg``。

    字段顺序：时间戳 → 等级 → logger 名 → 可选 stage.event 标签 →
    ContextVar 自动注入字段 → 记录 extras（去重）→ 消息 → 堆栈（多行缩进）。

    缺省字段不打印，避免噪音；带值的 ``extra`` 才会显示。

    堆栈渲染：异常 traceback 紧随日志行，单独放在 ``--- traceback ---`` /
    ``--- end ---`` 框架之间，确保一行搜索 ``error_type=`` 仍可定位所有错误。
    """

    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime",
    }

    def __init__(self) -> None:
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = record.levelname
        logger_name = record.name
        stage = getattr(record, "stage", None)
        event = getattr(record, "event", None)
        tag = ""
        if stage or event:
            tag = f"[{stage or '-'}·{event or '-'}]"

        # ContextVar 自动字段（不污染 record.__dict__，仅用于本次 format）
        try:
            from utils.request_context import current_context
            ctx = current_context()
        except Exception:
            ctx = {}

        extras: list[str] = []
        seen: set[str] = {"stage", "event"}
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key in seen or key.startswith("_"):
                continue
            if value is None or value == "":
                continue
            extras.append(f"{key}={_short_repr(value)}")
            seen.add(key)
        for key, value in ctx.items():
            if key in seen or key in {"stage", "event"}:
                continue
            if value is None or value == "":
                continue
            extras.append(f"{key}={_short_repr(value)}")
            seen.add(key)

        msg = record.getMessage()
        prefix = f"{ts}  {level:<7} {logger_name} {tag}"
        extras_str = " ".join(extras)
        if extras_str:
            head = f"{prefix} {extras_str}  \"{msg}\""
        else:
            head = f"{prefix} \"{msg}\""

        if record.exc_info:
            tb = self.formatException(record.exc_info)
            return f"{head}\n--- traceback ---\n{tb}--- end ---"
        return head


def _short_repr(value: Any) -> str:
    """把值压成单行字符串，避免日志被换行炸开。"""
    if isinstance(value, str):
        s = value.replace("\n", " ").replace("\r", " ")
        return s if len(s) <= 200 else s[:197] + "..."
    if isinstance(value, (list, tuple)):
        return f"[{','.join(_short_repr(v) for v in value[:6])}{'...' if len(value) > 6 else ''}]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}={_short_repr(v)}" for k, v in list(value.items())[:6]) + "}"
    return repr(value)


def setup_logging() -> None:
    """安装根 logger handler（控制台 + 可选文件），重复调用幂等。"""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = TextFormatter()
    root = logging.getLogger()
    root.setLevel(level)
    # 清掉默认 basicConfig handler，避免双写
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(level)
    root.addHandler(console)

    if os.getenv("LOG_TO_FILE", "true").lower() in ("1", "true", "yes"):
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "server.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    # 静默第三方噪音（保留 WARN+）
    for noisy in ("httpx", "sentence_transformers", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _CONFIGURED = True