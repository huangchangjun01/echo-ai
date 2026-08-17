"""启动入口：从 settings 读取 host/port。"""

from __future__ import annotations

import io
import sys

import uvicorn

from config.config import get_settings


def _setup_hf_env() -> None:
    """在任何 transformers / huggingface_hub 导入前固化 HF 镜像 / 缓存根。

    关键：huggingface_hub 在 import 时就把 ``HF_ENDPOINT`` 读进模块常量
    （``constants.ENDPOINT``），之后改 ``os.environ`` 无效。因此必须在
    ``uvicorn.run("app.agent_runner:app")`` 触发 huggingface 导入**之前**设置，
    否则缓存未命中时下载会直连 huggingface.co（被墙超时），而不是走 hf-mirror.com。
    """
    from utils.model_cache import apply_hf_env

    s = get_settings()
    apply_hf_env(s.embedding.endpoint, s.embedding.download_timeout)


def _force_utf8_stdio() -> None:
    """Windows 默认 cp936/GBK 控制台在写入 emoji/中文时会抛 UnicodeEncodeError。

    把 stdout/stderr 重新包成 UTF-8，避免启动后第 1 条带 emoji 的日志中断后续
    ``logging`` 输出（uvicorn 的内置 access log 会直接 print 到 stderr）。
    """
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        # 在非交互环境（无 buffer 属性）下静默跳过
        pass


def main() -> None:
    _force_utf8_stdio()
    _setup_hf_env()  # 必须早于 app.agent_runner 导入
    settings = get_settings().app
    host = settings.host
    port = settings.port

    if "pydevd" in sys.modules:
        # PyCharm Debug 模式：手动启动避免 loop_factory 冲突
        import asyncio

        config = uvicorn.Config("app.agent_runner:app", host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
    else:
        uvicorn.run("app.agent_runner:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()