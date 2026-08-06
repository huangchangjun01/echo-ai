"""read_memory_full：把某条记忆的完整 .md 内容拉到 LLM 上下文。

设计动机
========
`search_recall_for_chat` 只回 EchoRecall 里的 `summary` 字段（解析时抽出的摘要，约 200 字）。
当用户问题需要"细节"（如"那顿饭具体几号吃的？""拉师傅生日那天我都做了什么？"），
摘要往往不够。LLM 在 ReAct 中应能主动选择"展开某条记忆"，把整段 md 喂回自己。

走 echo-core 代理而非直连 Qiniu 的原因
======================================
- 鉴权：Qiniu 私有空间下载需服务端 AK/SK 签名，echo-core 已经持有；让 echo-core 转发可避免在
  echo-ai 重复持有 / 泄漏凭据。
- 一致性：echo-core `/api/memory/md-content` 优先读 MySQL `recall_memory.md_content` 缓存，
  缓存未命中再回源 Qiniu，绕开 CDN 421。
- 租户隔离：接口强制校验 `userId` 归属，前端/客户端无法伪造"别人的 memoryId"。

入参 / 出参
===========
- 入参：memory_id（必填，32 字符 UUID 去横线）、user_id（必填，租户标识）
- 出参：{"memory_id", "topic", "subjective_desc", "md_content", "md_len"} 或 ok=False
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx
from pydantic import Field

from tools.base import BaseTool, ToolResult, fail, ok
from utils.request_context import log_exception, merge_extra

logger = logging.getLogger(__name__)


# memoryId 是 32 位无横线 UUID；这里做轻校验，避免明显错误的输入打到 echo-core
_MEMORY_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")

# 拉回来的 md 默认截断到这个长度再交给 LLM；超出部分用 "[...截断 N 字符...]" 提示，
# 避免一条超长 md 直接吃光 LLM 上下文窗口
_MAX_MD_CHARS = 8000


class ReadMemoryFullTool(BaseTool):
    name = "read_memory_full"
    description = (
        "read_memory_full(memory_id: str, user_id: str) -> "
        "把指定 memory_id 对应的整段 .md 内容拉回 LLM 上下文。"
        "适用场景：用户问『那件事的细节』『当时的详细情况』『第几条记忆里到底写了什么』"
        "等需要展开摘要之外的细节的提问。"
        "memory_id 通常从 search_recall / EchoRecall 摘要列表中获得（形如 32 位 hex）。"
        "返回 {topic, subjective_desc, md_content, md_len}；md_content 已被截断以保护上下文。"
    )

    async def arun(
        self,
        memory_id: str = "",
        user_id: str = "",
        **_,
    ) -> ToolResult:
        return await _read_memory_full_async(memory_id=memory_id, user_id=user_id)

    def run(
        self,
        memory_id: str = "",
        user_id: str = "",
        **_,
    ) -> ToolResult:
        # 同步入口仅供离线 / 测试使用；ReAct 走 arun 异步路径
        import asyncio
        try:
            asyncio.get_running_loop()
            return fail("Use arun() in async context")
        except RuntimeError:
            return asyncio.run(_read_memory_full_async(memory_id=memory_id, user_id=user_id))


async def _read_memory_full_async(*, memory_id: str, user_id: str) -> ToolResult:
    if not memory_id or not _MEMORY_ID_RE.match(memory_id):
        return fail("invalid memory_id (expect 32-char hex)")
    if not user_id:
        return fail("user_id is required")

    base = (os.environ.get("ECHO_CORE_BASE_URL") or "http://localhost:8080").rstrip("/")
    token = os.environ.get("ECHO_CORE_INTERNAL_TOKEN") or ""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Internal-Token"] = token

    try:
        async with httpx.AsyncClient(timeout=60, headers=headers) as c:
            r = await c.post(
                f"{base}/api/memory/md-content",
                json={"userId": str(user_id), "memoryId": memory_id},
            )
    except Exception as e:
        log_exception(
            logger,
            "read_memory_full: echo-core request failed",
            exc=e,
            level=logging.WARNING,
            stage="tool_read_memory_full",
            event="http_error",
            memory_id=memory_id,
        )
        return fail(f"echo-core unreachable: {e}")

    if r.status_code == 404:
        # echo-core 在 md_content 未缓存时返回 404 + 中文 "md 内容未缓存（早期数据）"
        # 也可能 memoryId 不存在。这里统一 fail，让 LLM 知道"拿不到"，而不是空字符串误导。
        logger.info(
            "read_memory_full: echo-core 404 (memory missing or md not cached)",
            extra=merge_extra(
                stage="tool_read_memory_full",
                event="not_found",
                memory_id=memory_id,
                user_id=user_id,
            ),
        )
        return fail("memory not found or md not yet cached (parse still running?)")

    if r.status_code != 200:
        logger.warning(
            "read_memory_full: echo-core returned non-200",
            extra=merge_extra(
                stage="tool_read_memory_full",
                event="bad_status",
                memory_id=memory_id,
                status=r.status_code,
                body=r.text[:160],
            ),
        )
        return fail(f"echo-core returned {r.status_code}")

    # echo-core /api/memory/md-content 直接返回 text/markdown，body 是 md 全文
    md_full = r.content.decode("utf-8", errors="replace")
    md_len = len(md_full)

    # 截断保护 LLM 上下文
    truncated = False
    md = md_full
    if md_len > _MAX_MD_CHARS:
        md = md_full[:_MAX_MD_CHARS]
        truncated = True

    # 同时把 topic / subjective_desc 也带回来（先打 detail 拿元数据，省一次往返）
    topic = ""
    subjective_desc = ""
    try:
        # 复用一个无 head 的内部 client 拿 detail
        async with httpx.AsyncClient(timeout=15, headers=headers) as c2:
            d = await c2.get(
                f"{base}/api/memory/detail",
                params={"memoryId": memory_id},
                headers={"X-Session-Id": ""},  # detail 需要 X-Session-Id；此处用 X-Internal-Token 兜底
            )
            # echo-core /api/memory/detail 走 session 中间件；不带 session 会 401
            # 因此用 internal token 直连（仅当 token 配置了）需要 echo-core 暴露内部 detail 接口。
            # 当前 echo-core 没有 internal /api/memory/detail，所以这里降级：topic/desc 留空，
            # 让 LLM 仅凭 md_content 也能识别主题（md 第一行通常是 "# {topic}"）
        # 如果未来 echo-core 暴露 /api/memory/internal/detail，这里可以补回 topic/subjective_desc
    except Exception:
        # 元数据拿不到不影响主功能
        pass

    # 从 md 头里解析出 topic（"# xxx" 形式）作为兜底
    if not topic:
        first_line = md_full.split("\n", 1)[0].strip()
        if first_line.startswith("# "):
            topic = first_line[2:].strip()

    payload: dict[str, Any] = {
        "memory_id": memory_id,
        "topic": topic,
        "subjective_desc": subjective_desc,
        "md_content": md,
        "md_len": md_len,
        "truncated": truncated,
    }
    logger.info(
        "read_memory_full ok",
        extra=merge_extra(
            stage="tool_read_memory_full",
            event="ok",
            memory_id=memory_id,
            user_id=user_id,
            md_len=md_len,
            truncated=truncated,
        ),
    )
    return ok(payload)
