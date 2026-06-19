import sys

import uvicorn

if __name__ == "__main__":
    # 检测是否在 PyCharm Debug 模式下
    is_debug = 'pydevd' in sys.modules

    if is_debug:
        # Debug 模式：手动启动，避免 loop_factory 冲突
        import asyncio

        config = uvicorn.Config("app.agent_runner:app", host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        asyncio.run(server.serve())  # 不传递 loop_factory
    else:
        # 普通模式：正常启动
        # Run the LangChain-based agent runner
        uvicorn.run("app.agent_runner:app", host="0.0.0.0", port=8000, reload=False)
