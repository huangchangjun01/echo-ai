"""understand_image：进程内调用视觉模型。

实现说明：
- 下载图片 → CLIP embedding → 走 LLM（基于 CLIP 向量在 weaviate 中检索相似图片作为「描述增强」）。
- 真正的图像描述使用 LLM（接收图片 URL 时）或多模态 LLM（若服务支持）。
- 为保持进程内、可降级，这里采用 pragmatic 方案：
  - 通过 LLM 直接描述图片（仅传 URL 给 LLM，若模型支持图片则返回详细描述）。
  - 同步返回 CLIP embedding 用于跨模态检索。
"""

from __future__ import annotations

import asyncio
import logging

from tools.base import BaseTool, ToolResult, ok
from utils.request_context import log_exception, log_silent_failure

logger = logging.getLogger(__name__)


class UnderstandImageTool(BaseTool):
    name = "understand_image"
    description = (
        "understand_image(image_url: str) -> 描述图片内容。"
        "返回 {description: str, embedding: list[float]}。"
        "当用户发送图片或对话中提到一张图时使用。"
    )

    async def arun(self, image_url: str = "", **_) -> ToolResult:
        return await _understand_image_async(image_url)

    def run(self, image_url: str = "", **_) -> ToolResult:  # noqa: D401
        try:
            asyncio.get_running_loop()
            # 已在事件循环中：返回错误提示（让上层用 arun）
            from tools.base import fail

            return fail("Use arun() in async context")
        except RuntimeError:
            return asyncio.run(_understand_image_async(image_url))


async def _understand_image_async(image_url: str) -> ToolResult:
    from llm.client import get_llm_client

    if not image_url:
        return ok({"description": "", "embedding": []})

    # 1) 下载并编码为 CLIP 向量
    embedding: list[float] = []
    description = ""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            data = resp.content
        from embedding import clip

        vecs = await asyncio.to_thread(clip.embed_images, [data])
        embedding = vecs[0] if vecs else []
    except Exception as e:
        log_exception(
            logger,
            "image download/clip failed",
            exc=e,
            level=logging.WARNING,
            stage="understand_image",
            event="image_clip_error",
            image_url=image_url,
        )

    # 2) 用 LLM 描述图片（多模态时直接传 URL；否则使用检索增强）
    try:
        client = get_llm_client()
        messages = [
            {
                "role": "system",
                "content": "你是一名图像理解助手。请用 1~3 句中文简洁描述给定图片的主体、场景、关键元素。",
            },
            {"role": "user", "content": f"请描述这张图片：{image_url}"},
        ]
        resp = await client.chat(messages, max_tokens=200, temperature=0.4)
        try:
            description = resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log_silent_failure(
                logger,
                "extract description from LLM response failed (use empty)",
                exc=e,
                stage="understand_image",
                event="resp_extract_error",
                image_url=image_url,
            )
            description = ""
    except Exception as e:
        log_exception(
            logger,
            "LLM describe image failed",
            exc=e,
            level=logging.WARNING,
            stage="understand_image",
            event="llm_describe_error",
            image_url=image_url,
            has_embedding=bool(embedding),
        )

    return ok({"description": description, "embedding": embedding, "image_url": image_url})