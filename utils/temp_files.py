"""临时文件上下文：解析器下载/处理/清理统一封装。

每个解析调用 `temp_workspace()` 拿到一个工作目录，用完自动清理，
避免大视频/音频残留占用磁盘。
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def temp_workspace(prefix: str = "echo-recall-") -> Iterator[Path]:
    """yield 一个临时目录，with 退出时强制清理（含子文件）。"""
    base = tempfile.mkdtemp(prefix=prefix)
    base_path = Path(base)
    try:
        yield base_path
    finally:
        try:
            shutil.rmtree(base, ignore_errors=True)
            logger.debug("temp workspace cleaned: %s", base)
        except Exception as e:
            logger.debug("temp workspace cleanup failed (ignored): %s | %s", base, e)


def safe_remove(path: Path | str | None) -> None:
    """尽力删除单文件/目录，吞掉所有异常（cleanup 路径专用）。"""
    if path is None:
        return
    try:
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink(missing_ok=True)
    except Exception as e:
        logger.debug("safe_remove ignored: %s | %s", path, e)