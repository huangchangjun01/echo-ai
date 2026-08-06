"""回忆记忆业务编排：

- parse_memory：解析多模态源文件 → LLM 总结 → 生成 md → 直传对象存储 → 写 EchoRecall
- reparse_memory：编辑已有记忆后，先清旧向量再走 parse_memory 完整链路（独立 stage 便于排查）
- delete_source：删某文件对应片段 → 重算摘要 → 覆盖原 md → 更新向量
- delete_memory：删 EchoRecall 中的向量记录（对象存储与 MySQL 由 echo-core 处理）

设计要点：
- 解析链路较慢（多模态下载 + LLM），用 fire-and-forget 异步触发。
- 编辑锁防并发写入：parse_memory / reparse_memory / delete_source 都会尝试抢锁。
- 临时文件统一通过 utils.temp_files.temp_workspace 清理。
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from database.recall_status import mark_done, release_edit_lock, try_acquire_edit_lock, update_status
from embedding import bge_m3
from llm.client import get_llm_client
from parsers.base import ParsedFile
from parsers.registry import parse_file
from skills import build_memory_md, parse_summary, remove_source_section, update_source_metadata
from storage.qiniu_client import delete_object, upload_bytes
from utils.request_context import log_exception, merge_extra
from vector.recall_store import get_recall_store

logger = logging.getLogger(__name__)

_PARSE_STATUS_PENDING = 0
_PARSE_STATUS_RUNNING = 1
_PARSE_STATUS_DONE = 2
_PARSE_STATUS_FAILED = 3

# 单个源文件解析产物喂给 build_memory_md 的上限（防御性深度第一道兜底）：
# - 修复视频重解析 bug：MiniMax-M3（小模型）上下文窗口紧，单个 8 帧视频 detail 合并后
#   可达 12K+ 字符，触发了 "context window exceeds limit" 400 错误。
# - 这里再次兜底限长，避免任何一个上游 parser（video/audio/text/image）输出异常大段内容。
_MAX_PER_SOURCE_DETAIL_CHARS = 4000

# 防御性 HTML 剥离（万一上游解析器把 HTML 串进了 detail 内容，落盘前必须清洗）
_HTML_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)[\s\S]*?</\1>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&(?:nbsp|amp|lt|gt|quot|#39);", re.IGNORECASE)


def _strip_html(text: str) -> str:
    """去掉残留 HTML：先剥 script/style 整块，再剥普通标签，最后解常见实体。"""
    if not text:
        return text
    s = _HTML_SCRIPT_STYLE_RE.sub("", text)
    s = _HTML_TAG_RE.sub("", s)
    s = _HTML_ENTITY_RE.sub(lambda m: {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'"}.get(m.group(0).lower(), m.group(0)), s)
    # 多余空行折叠
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ===== 公共：解析单个源文件 =====

async def _parse_one(src: dict[str, Any]) -> tuple[str, ParsedFile]:
    file_key = src.get("fileKey") or ""
    file_name = src.get("fileName") or "unknown"
    file_type = int(src.get("fileType") or 0)
    url = src.get("url") or ""
    try:
        # 关键：透传真实 fileKey（此前误传 fileName，导致解析器无法直连七牛，
        # 只能退回前端 url，把 SPA fallback/HTML 当成源文件内容写进 md）
        parsed = await parse_file(file_key, file_name, file_type, url)
        return file_name, parsed
    except Exception as e:
        log_exception(
            logger,
            "parse one failed",
            exc=e,
            level=logging.WARNING,
            stage="recall_parse_one",
            event="error",
            file_name=file_name,
        )
        return file_name, ParsedFile(modality="text", meta={"error": str(e)[:200]})


def _is_failed_parse(p: ParsedFile) -> bool:
    """判定一个 ParsedFile 是否解析失败（无文本、无 detail_md）。"""
    if p.meta.get("error"):
        return True
    if not (p.text or p.detail_md):
        return True
    return False


async def _parse_all(sources: list[dict[str, Any]]) -> list[tuple[str, ParsedFile]]:
    """并发解析所有源文件（受并发上限约束，防止大文件/多模态过载）。"""
    sem = asyncio.Semaphore(4)

    async def _bounded(src: dict[str, Any]) -> tuple[str, ParsedFile]:
        async with sem:
            return await _parse_one(src)

    return await asyncio.gather(*[_bounded(s) for s in sources])


def _md_key(user_id: str, role_id: str, memory_id: str) -> str:
    return f"memory/{user_id}/{role_id}/{memory_id}/{memory_id}.md"


# ===== parse_memory =====

async def parse_memory(
    user_id: str,
    role_id: str,
    memory_id: str,
    topic: str,
    subjective_desc: str,
    sources: list[dict[str, Any]],
) -> None:
    """解析记忆：抢锁 → 解析 → 生成 md → 上传 → 写向量 → 解锁。"""
    role_id = role_id or "default"
    md_key = _md_key(user_id, role_id, memory_id)
    await mark_done(memory_id, parse_status=_PARSE_STATUS_RUNNING) if False else None  # 实际只更新 parse

    # 抢锁（AI 写入中）
    if not await try_acquire_edit_lock(memory_id):
        log_exception(
            logger,
            "acquire edit lock timeout",
            exc=None,
            level=logging.WARNING,
            stage="recall_parse",
            event="lock_timeout",
            memory_id=memory_id,
        )
        await mark_done(memory_id, parse_status=_PARSE_STATUS_FAILED)
        return

    try:
        # 1) 并发解析所有源
        results = await _parse_all(sources)
        details: list[dict] = []
        failed_sources: list[str] = []
        for name, p in results:
            # 视频走"场景分段"路径：parser 把每个场景作为独立 chunk 给出，
            # 业务层负责展开成 N 个 details[] 条目，自然渲染成 N 个 `### 片段`。
            # 这种路径不再依赖 detail_md（避免与 chunks 重复生成内容）。
            if p.modality == "video" and p.chunks:
                any_ok = False
                logger.info(
                    "parse_memory: video scene fan-out",
                    extra=merge_extra(
                        stage="recall_parse",
                        event="scene_fanout_start",
                        memory_id=memory_id,
                        file_name=name,
                        chunk_count=len(p.chunks),
                    ),
                )
                # 视频走"1 detail per source"策略：N 个场景打包成 1 个 detail，
                # 所有结构化数据 + 渲染文本挂在 detail.meta.video_scenes_structured 里。
                # 这样 build_memory_md 的 _generate_segment_section 只调 1 次，
                # 渲染"贯穿主体 + N 个 ### 片段"整体；避免老路径 LLM 把短 prose
                # 重新展开成 500+ 字的重复描述。
                scene_structured: list[dict] = []
                joined_prose_parts: list[str] = []
                for ci, ch in enumerate(p.chunks, 1):
                    text = _strip_html(ch.text or "")
                    if not text:
                        continue
                    any_ok = True
                    # 透传结构化数据
                    chunk_meta = getattr(ch, "meta", None) or {}
                    scene_structured.append({
                        "idx": ci,
                        "structured": bool(chunk_meta.get("structured")),
                        "characters": chunk_meta.get("characters", []),
                        "setting": chunk_meta.get("setting", ""),
                        "actions": chunk_meta.get("actions", []),
                        "objects": chunk_meta.get("objects", []),
                        "text": chunk_meta.get("text", ""),
                        "expressions": chunk_meta.get("expressions", []),
                        "color_spatial": chunk_meta.get("color_spatial", ""),
                        "deltas": chunk_meta.get("deltas", "初始"),
                        "prose": chunk_meta.get("prose", text),
                    })
                    joined_prose_parts.append(text)
                # 1 个 detail 承载整个视频
                if scene_structured:
                    joined = "\n\n".join(joined_prose_parts)
                    if len(joined) > _MAX_PER_SOURCE_DETAIL_CHARS:
                        joined = joined[:_MAX_PER_SOURCE_DETAIL_CHARS]
                    detail_entry: dict = {"fileName": name, "detail": joined}
                    if any(s.get("structured") for s in scene_structured):
                        detail_entry["meta"] = {
                            "video_scenes_structured": scene_structured,
                        }
                    details.append(detail_entry)
                logger.info(
                    "parse_memory: video scene fan-out done",
                    extra=merge_extra(
                        stage="recall_parse",
                        event="scene_fanout_done",
                        memory_id=memory_id,
                        file_name=name,
                        details_added=len(details),
                        scenes=len(scene_structured),
                    ),
                )
                if not any_ok:
                    failed_sources.append(name)
                continue
            if p.detail_md:
                # 兜底：剥掉上游可能串入的 HTML 标签（如 LLM 把 SPA fallback 当图像描述）
                clean_detail = _strip_html(p.detail_md)
                if clean_detail:
                    # 防御性深度：单源 detail 长度上限，避免拼到 build_memory_md 的
                    # user prompt 里超过 MiniMax-M3 的上下文窗口。
                    capped = (
                        clean_detail[:_MAX_PER_SOURCE_DETAIL_CHARS]
                        if len(clean_detail) > _MAX_PER_SOURCE_DETAIL_CHARS
                        else clean_detail
                    )
                    details.append({"fileName": name, "detail": capped})
                else:
                    failed_sources.append(name)
            elif _is_failed_parse(p):
                failed_sources.append(name)

        # 当所有源都解析失败时，**绝不落盘**：空壳 md 会污染用户记忆库。
        # 直接标 FAILED，让前端提示用户重试或检查源文件。
        if sources and not details:
            err_summary = "; ".join(failed_sources[:3])
            logger.warning(
                "parse_memory: all sources failed, skip md write",
                extra=merge_extra(
                    stage="recall_parse",
                    event="all_failed",
                    memory_id=memory_id,
                    sources=len(sources),
                    failed=len(failed_sources),
                    failed_sample=err_summary[:300],
                ),
            )
            await mark_done(memory_id, parse_status=_PARSE_STATUS_FAILED)
            return

        # 部分失败：在 md 元数据里追加"解析失败"列表，方便用户定位问题
        if failed_sources:
            details.append(
                {
                    "fileName": "解析失败清单",
                    "detail": "以下文件解析失败，已跳过：\n"
                    + "\n".join(f"- {n}" for n in failed_sources),
                }
            )

        source_files = [name for name, _ in results]

        # 2) 生成 md
        md = await build_memory_md(
            topic=topic,
            subjective_desc=subjective_desc or "",
            details=details,
            source_files=source_files,
            use_large_model=len(details) > 3,
        )

        # 3) 上传 md 到对象存储 + 同步缓存到 echo-core 数据库（避免后续 421 下载失败）
        upload_bytes(md_key, md.encode("utf-8"), mime_type="text/markdown; charset=utf-8")
        await _save_md_to_core(user_id, memory_id, md)

        # 4) 写 EchoRecall（摘要向量）
        summary = parse_summary(md) or md[:200]
        if summary:
            try:
                vecs = await asyncio.to_thread(bge_m3.embed_texts, [summary])
                vec = vecs[0] if vecs else []
                if vec:
                    store = get_recall_store()
                    store.upsert(
                        memory_id=memory_id,
                        summary=summary,
                        user_id=user_id,
                        role_id=role_id,
                        md_key=md_key,
                        topic=topic,
                        embedding=vec,
                        intensity=0.7,
                    )
            except Exception as e:
                log_exception(
                    logger,
                    "recall vector upsert failed (non-fatal)",
                    exc=e,
                    level=logging.WARNING,
                    stage="recall_vector",
                    event="upsert_error",
                    memory_id=memory_id,
                )

        # 5) 标记完成
        await mark_done(memory_id, parse_status=_PARSE_STATUS_DONE)
        logger.info(
            "parse_memory ok",
            extra=merge_extra(
                stage="recall_parse",
                event="ok",
                memory_id=memory_id,
                sources=len(sources),
                md_len=len(md),
            ),
        )
    except Exception as e:
        log_exception(
            logger,
            "parse_memory failed",
            exc=e,
            level=logging.ERROR,
            stage="recall_parse",
            event="error",
            memory_id=memory_id,
        )
        await mark_done(memory_id, parse_status=_PARSE_STATUS_FAILED)
    finally:
        await release_edit_lock(memory_id)


# ===== reparse_memory =====

async def reparse_memory(
    user_id: str,
    role_id: str,
    memory_id: str,
    topic: str,
    subjective_desc: str,
    sources: list[dict[str, Any]],
    removed_file_keys: list[str] | None = None,
) -> None:
    """编辑已有记忆后重新解析。

    与 parse_memory 的关键差异：
    1. 进入前**先尝试清旧向量**，避免在上游解析失败时残留旧数据；
       即使 RecallVectorStore.upsert 已经是"先删后插"的幂等实现，显式删除
       也让"清干净"的语义更清晰，且便于排查"向量没刷新"这类问题。
    2. 走独立 stage `recall_reparse`，日志与指标与"首次创建"区分开来。
    3. 不在本函数处理 removed_file_keys 的 md 片段删除——这一逻辑由
       echo-core 在事务提交阶段调用 /memory/source/delete 链路完成
       （参见 echo-core.DeleteSourceFile 软删后异步调用 delete_source）。
       若上游确保已经先 /source/delete 过，本函数只需全量重生。
    """
    role_id = role_id or "default"
    logger.info(
        "reparse_memory start",
        extra=merge_extra(
            stage="recall_reparse",
            event="start",
            memory_id=memory_id,
            sources=len(sources),
            removed=len(removed_file_keys or []),
        ),
    )

    # 1) 先删旧向量（best-effort；不阻塞后续解析）
    try:
        store = get_recall_store()
        store.delete_by_memory_id(user_id, memory_id)
    except Exception as e:
        log_exception(
            logger,
            "reparse_memory: pre-delete vector failed (non-fatal)",
            exc=e,
            level=logging.WARNING,
            stage="recall_reparse",
            event="pre_delete_vector_error",
            memory_id=memory_id,
        )

    # 2) 把 parse_status 推到 RUNNING（幂等；parse_memory 自己也会再触发但显式更清晰）。
    #    注意：用 update_status 而非 mark_done——mark_done 会同时把 edit_status 重置为 0，
    #    可能窃取其他并发流程的编辑锁。
    try:
        await update_status(memory_id, parse_status=_PARSE_STATUS_RUNNING)
    except Exception as e:
        log_exception(
            logger,
            "reparse_memory: mark RUNNING failed (non-fatal)",
            exc=e,
            level=logging.WARNING,
            stage="recall_reparse",
            event="mark_running_error",
            memory_id=memory_id,
        )

    # 3) 全量重建：走标准 parse_memory 链路（抢锁 → 解析 → md → 向量 → 解锁）
    #    parse_memory 内部的 upsert 会再做一次 stable-id 维度的清 + 写，双保险。
    await parse_memory(
        user_id=user_id,
        role_id=role_id,
        memory_id=memory_id,
        topic=topic,
        subjective_desc=subjective_desc,
        sources=sources,
    )
    logger.info(
        "reparse_memory dispatched to parse_memory",
        extra=merge_extra(
            stage="recall_reparse",
            event="dispatched",
            memory_id=memory_id,
            sources=len(sources),
        ),
    )


# ===== delete_source =====

async def delete_source(
    user_id: str,
    role_id: str,
    memory_id: str,
    file_key: str,
    file_name: str,
) -> None:
    """删某个源文件：尝试从 md 中删除对应片段 → 覆盖 md → 更新向量。

    注意：本函数假定对象存储中 file_key 已被 echo-core 删过；本函数只动 md 与向量。
    设计取舍：Qiniu CDN 对私有空间下载在某些环境会返回 421（misdirected request，
    virtual host 路由问题，与 Python SDK 签名无关）。这里采取"尽力重写 + 失败兜底
    不阻塞"——md 片段残留不会影响核心功能，向量摘要会被更新一次（取当前 md 的摘要，
    即使包含残留片段，相似度阈值会过滤掉）。
    """
    role_id = role_id or "default"
    md_key = _md_key(user_id, role_id, memory_id)
    if not await try_acquire_edit_lock(memory_id):
        logger.warning("delete_source: lock timeout memory_id=%s", memory_id)
        return
    try:
        # 1) 下载原 md（经 echo-core 代理，绕开 Qiniu CDN 421）
        md: str = ""
        try:
            md = await _fetch_md_content(user_id, memory_id)
        except Exception as e:
            log_exception(
                logger,
                "delete_source: download md failed (will skip md rewrite, keep vector update)",
                exc=e,
                level=logging.WARNING,
                stage="recall_delete_source",
                event="download_error",
                memory_id=memory_id,
            )

        # 2) 删除该文件片段 & 更新元数据 来源
        new_md = md
        if md:
            new_md = remove_source_section(md, file_name)
            import re
            m = re.search(r"(- 来源:\s*\[)([^\]]*)(\])", new_md)
            remaining = []
            if m:
                remaining = [x.strip() for x in m.group(2).split(",") if x.strip() and x.strip() != file_name]
            new_md = update_source_metadata(new_md, file_name, remaining)

        # 3) 覆盖上传 md（仅在拿到 md 时才覆盖） + 同步缓存到 echo-core 数据库
        if new_md and md:
            try:
                upload_bytes(md_key, new_md.encode("utf-8"), mime_type="text/markdown; charset=utf-8")
                await _save_md_to_core(user_id, memory_id, new_md)
            except Exception as e:
                log_exception(
                    logger,
                    "delete_source: upload/save md failed (non-fatal)",
                    exc=e,
                    level=logging.WARNING,
                    stage="recall_delete_source",
                    event="upload_error",
                )

        # 4) 更新 EchoRecall 摘要（取更新后的摘要；若 md 重写失败则用原摘要）
        new_summary = parse_summary(new_md) if new_md else parse_summary(md)
        if new_summary:
            try:
                vecs = await asyncio.to_thread(bge_m3.embed_texts, [new_summary])
                vec = vecs[0] if vecs else []
                if vec:
                    store = get_recall_store()
                    store.upsert(
                        memory_id=memory_id,
                        summary=new_summary,
                        user_id=user_id,
                        role_id=role_id,
                        md_key=md_key,
                        topic="(updated)",
                        embedding=vec,
                        intensity=0.6,
                    )
            except Exception as e:
                log_exception(
                    logger,
                    "delete_source vector upsert failed",
                    exc=e,
                    level=logging.WARNING,
                    stage="recall_delete_source",
                    event="vector_error",
                )

        logger.info(
            "delete_source ok",
            extra=merge_extra(stage="recall_delete_source", event="ok", memory_id=memory_id, file_name=file_name, md_rewritten=bool(md)),
        )
    finally:
        await release_edit_lock(memory_id)


# ===== delete_memory =====


async def _fetch_md_url(user_id: str, memory_id: str) -> str:
    """通过 echo-core 拿 md 私有下载 URL（保留兼容老逻辑）。

    推荐使用 _fetch_md_content 直接拿内容，能绕过 Qiniu CDN 域名的 421。
    """
    import os

    import httpx

    base = (os.environ.get("ECHO_CORE_BASE_URL") or "http://localhost:8080").rstrip("/")
    token = os.environ.get("ECHO_CORE_INTERNAL_TOKEN") or ""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Internal-Token"] = token
    try:
        async with httpx.AsyncClient(timeout=10, headers=headers) as c:
            r = await c.post(f"{base}/api/memory/md-url", json={"userId": user_id, "memoryId": memory_id})
            if r.status_code != 200:
                logger.warning("_fetch_md_url: echo-core returned %d", r.status_code)
                return ""
            d = r.json()
            return (d.get("data") or {}).get("url") or ""
    except Exception as e:
        log_exception(
            logger,
            "_fetch_md_url failed",
            exc=e,
            level=logging.WARNING,
            stage="recall_fetch_md_url",
            event="error",
            memory_id=memory_id,
        )
        return ""


async def _fetch_md_content(user_id: str, memory_id: str) -> str:
    """通过 echo-core 代理端点直接拿 md 内容（推荐路径，绕开 CDN 421）。

    echo-core 的 /api/memory/md-content 直接从 DB 读 md_content 字段，永远可达。
    """
    import os

    import httpx

    base = (os.environ.get("ECHO_CORE_BASE_URL") or "http://localhost:8080").rstrip("/")
    token = os.environ.get("ECHO_CORE_INTERNAL_TOKEN") or ""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Internal-Token"] = token
    try:
        async with httpx.AsyncClient(timeout=60, headers=headers) as c:
            r = await c.post(f"{base}/api/memory/md-content", json={"userId": user_id, "memoryId": memory_id})
            if r.status_code != 200:
                logger.warning("_fetch_md_content: echo-core returned %d", r.status_code)
                return ""
            return r.content.decode("utf-8", errors="replace")
    except Exception as e:
        log_exception(
            logger,
            "_fetch_md_content failed",
            exc=e,
            level=logging.WARNING,
            stage="recall_fetch_md_content",
            event="error",
            memory_id=memory_id,
        )
        return ""


async def _save_md_to_core(user_id: str, memory_id: str, md: str) -> None:
    """把 md 内容缓存到 echo-core 数据库（md_content 字段）。

    解决 Qiniu CDN 偶尔 421 Misdirected Request 导致 echo-ai 无法下载 md 的问题。
    echo-ai 上传 md 到对象存储后立刻同步缓存到 echo-core DB，后续从 DB 读，永远可达。
    """
    import os

    import httpx

    base = (os.environ.get("ECHO_CORE_BASE_URL") or "http://localhost:8080").rstrip("/")
    token = os.environ.get("ECHO_CORE_INTERNAL_TOKEN") or ""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Internal-Token"] = token
    try:
        async with httpx.AsyncClient(timeout=30, headers=headers) as c:
            r = await c.post(f"{base}/api/memory/md-content/save", json={
                "memoryId": memory_id,
                "mdContent": md,
            })
            if r.status_code != 200:
                logger.warning("_save_md_to_core: echo-core returned %d", r.status_code)
                return
        logger.info(
            "_save_md_to_core ok",
            extra=merge_extra(stage="recall_save_md", event="ok", memory_id=memory_id, bytes=len(md)),
        )
    except Exception as e:
        log_exception(
            logger,
            "_save_md_to_core failed (non-fatal)",
            exc=e,
            level=logging.WARNING,
            stage="recall_save_md",
            event="error",
            memory_id=memory_id,
        )

async def delete_memory(user_id: str, role_id: str, memory_id: str) -> None:
    """删 EchoRecall 向量。对象存储与 MySQL 由 echo-core 处理。"""
    try:
        n = get_recall_store().delete_by_memory_id(user_id, memory_id)
        logger.info(
            "delete_memory ok",
            extra=merge_extra(stage="recall_delete_memory", event="ok", memory_id=memory_id, affected=n),
        )
    except Exception as e:
        log_exception(
            logger,
            "delete_memory failed (non-fatal)",
            exc=e,
            level=logging.WARNING,
            stage="recall_delete_memory",
            event="error",
            memory_id=memory_id,
        )