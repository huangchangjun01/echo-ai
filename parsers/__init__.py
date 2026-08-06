"""Parsers package."""

from __future__ import annotations

import logging

# 子模块共用一个具名 logger，便于上游按 __name__ 过滤；不可用时回退到 root。
try:
    _logger = logging.getLogger("echo-ai.parsers")
    if not _logger.handlers and not _logger.parent.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        _logger.addHandler(_h)
        _logger.setLevel(logging.INFO)
except Exception:  # noqa: BLE001
    _logger = logging
