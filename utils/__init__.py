"""utils 工具包：下载器、工具集、日志设施、请求上下文。"""

from .downloader import DownloadError, download_file, download_file_async
from .logging_setup import (
    TextFormatter,
    get_logger,
    is_configured,
    setup_logging,
)
from .request_context import (
    bind_request,
    current_context,
    log_event,
    log_stage,
    merge_extra,
    request_context_scope,
    to_thread_with_ctx,
    unbind_request,
)
from .tools import build_vector_search_tool, tool_schemas

__all__ = [
    "DownloadError",
    "download_file",
    "download_file_async",
    "TextFormatter",
    "setup_logging",
    "is_configured",
    "get_logger",
    "bind_request",
    "current_context",
    "merge_extra",
    "log_event",
    "log_stage",
    "request_context_scope",
    "to_thread_with_ctx",
    "unbind_request",
    "build_vector_search_tool",
    "tool_schemas",
]
