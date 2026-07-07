"""understand_audio：进程内调用语音模型（Whisper 转录 + 嵌入）。

- 下载音频字节 → whisper 转文本 → BGE-M3 编码作为跨模态向量。
- 失败时返回空文本与零向量，工具调用方不应崩溃。
"""

from __future__ import annotations

import asyncio
import logging

from tools.base import BaseTool, ToolResult, ok

logger = logging.getLogger(__name__)


class UnderstandAudioTool(BaseTool):
    name = "understand_audio"
    description = (
        "understand_audio(audio_url: str) -> 转录音频并返回文本与声纹嵌入。"
        "返回 {text: str, embedding: list[float]}。"
        "当用户发送语音消息时使用。"
    )

    async def arun(self, audio_url: str = "", **_) -> ToolResult:
        return await _understand_audio_async(audio_url)

    def run(self, audio_url: str = "", **_) -> ToolResult:  # noqa: D401
        try:
            asyncio.get_running_loop()
            from tools.base import fail

            return fail("Use arun() in async context")
        except RuntimeError:
            return asyncio.run(_understand_audio_async(audio_url))


async def _understand_audio_async(audio_url: str) -> ToolResult:
    if not audio_url:
        return ok({"text": "", "embedding": []})

    try:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(audio_url)
            resp.raise_for_status()
            data = resp.content
    except Exception as e:
        logger.warning("audio download failed: %s", e)
        return ok({"text": "", "embedding": [], "error": str(e)})

    try:
        from embedding import whisper

        result = await asyncio.to_thread(whisper.embed_audio, data)
    except Exception as e:
        logger.warning("whisper embed failed: %s", e)
        return ok({"text": "", "embedding": [], "error": str(e)})

    return ok(result)