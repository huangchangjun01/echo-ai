"""启动入口：从 settings 读取 host/port。"""

from __future__ import annotations

import sys

import uvicorn

from config.config import get_settings


def main() -> None:
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