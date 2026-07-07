"""Chat 业务逻辑：预注入 + 轻量 ReAct + 流式级联 + 异步记忆抽取。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from config.config import get_settings
from config.prompts import build_system_prompt
from llm.cascade import _strip_think, cascade_chat
from llm.client import get_llm_client, parse_tool_call
from llm.intent import Intent, classify_intent
from memory import build_chat_context, extract_and_archive_async
from memory.retriever import causal_chain, load_persona
from tools import adispatch as dispatch_tool
from utils.request_context import log_stage, merge_extra

logger = logging.getLogger(__name__)


# ---------- 资源提取：把命中里的可下载 URL 抽成前端可渲染的 resource 事件 ----------

# 视为可渲染附件的模态；memory / text 仅作文本注入，不在前端显示缩略图。
_ATTACHMENT_MODALITIES = {"image", "audio", "video", "file"}

# 模态 → 默认 MIME（缺扩展名或缺元数据时兜底）
_DEFAULT_MIME_BY_MODALITY = {
    "image": "image/jpeg",
    "audio": "audio/mpeg",
    "video": "video/mp4",
    "file": "application/octet-stream",
}

# 文件扩展名 → MIME（用于按 URL 末端推断 MIME）
_EXT_TO_MIME: tuple[tuple[str, str], ...] = (
    (".jpg", "image/jpeg"),
    (".jpeg", "image/jpeg"),
    (".png", "image/png"),
    (".gif", "image/gif"),
    (".webp", "image/webp"),
    (".bmp", "image/bmp"),
    (".svg", "image/svg+xml"),
    (".mp3", "audio/mpeg"),
    (".m4a", "audio/mp4"),
    (".wav", "audio/wav"),
    (".ogg", "audio/ogg"),
    (".flac", "audio/flac"),
    (".mp4", "video/mp4"),
    (".webm", "video/webm"),
    (".mov", "video/quicktime"),
    (".pdf", "application/pdf"),
    (".json", "application/json"),
    (".txt", "text/plain"),
    (".md", "text/markdown"),
    (".csv", "text/csv"),
    (".log", "text/x-log"),
    (".py", "text/x-python"),
    (".zip", "application/zip"),
)


def _infer_mime_type(modality: str, url: str, md_mime: str | None = None) -> str:
    """三层降级推断 MIME：元数据显式字段 → URL 扩展名 → 模态兜底。"""
    if md_mime and "/" in md_mime:
        return md_mime
    if url:
        lower = url.lower().split("?", 1)[0].split("#", 1)[0]
        for ext, mime in _EXT_TO_MIME:
            if lower.endswith(ext):
                return mime
    return _DEFAULT_MIME_BY_MODALITY.get(modality, "application/octet-stream")


def _clean_filename(name: str) -> str:
    """去掉路径成分，仅保留文件名；截到 200 字。"""
    if not name:
        return "未命名"
    cleaned = str(name).replace("\\", "/").rstrip("/").split("/")[-1].strip()
    return (cleaned or "未命名")[:200]


def _normalize_url(url: str) -> str:
    """去掉 sourceUrl 的 http(s) 前缀。

    上下文：data 里的 sourceUrl 可能是裸域名（如 ``cdn.example.com/foo.jpg``），
    也可能带 https。前端要求不带 scheme —— 由前端根据运行环境自行拼接。
    """
    if not url:
        return ""
    s = url.strip()
    for prefix in ("https://", "http://", "//"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break
    return s


# 把一段文本里所有形如 ``https://...`` / ``http://...`` 的 URL 去掉 scheme 前缀。
# LLM 看到裸 URL 时会"自作主张"补 https://，因此必须在 final_text 出炉后再清一遍。
_URL_SCHEME_RE = re.compile(r"(?<![\w@/])https?://", re.IGNORECASE)


def _strip_url_schemes(text: str) -> str:
    if not text:
        return text or ""
    return _URL_SCHEME_RE.sub("", text)


def _hit_to_resource(hit: dict, *, source_tag: str, iter_no: int | None) -> dict | None:
    """从一条 hit 里抽出 resource 事件字段。无法形成 resource 时返回 None。

    字段说明（前端使用约定）：
    - event_id:  本次 yield 的唯一 id，用作 React key / 流内去重 key。
    - url:       无 scheme 的下载路径（前端按运行环拼接 http/https）。
    - name:      原始 fileName，与 markdown ``[name](url)`` 中的 label 一致。
    - display_name: 清洗后的纯文件名（去掉路径前缀），用于 UI 标题/悬停提示。
    - file_id:   元数据里的 fileId，可用于后端二次查询或点击行为埋点。
    - modality / mime_type: 模态与 MIME；模态走语义分支，MIME 用于 DOM/下载决策。
    - chunk_index / total_chunks: 多 chunk 文本片段的分片定位（图像通常 0/1）。
    - size_bytes: 文件大小（若有），无元数据时为 None。
    - similarity: 0~1 浮点，4 位小数。
    - source / iter: 触发源（l1_hint / search_memory / understand_image...）与 ReAct 迭代号。
    """
    md = hit.get("metadata") or {}
    url = _normalize_url(md.get("sourceUrl") or "")
    if not url:
        return None
    modality = (hit.get("modality") or md.get("modality") or "file").lower()
    if modality not in _ATTACHMENT_MODALITIES:
        return None
    raw_name = (
        md.get("fileName")
        or hit.get("content")
        or md.get("fileId")
        or "未命名"
    )
    name = (str(raw_name).strip() or "未命名")[:200]
    file_id = str(md.get("fileId") or md.get("file_id") or hit.get("id") or "")
    try:
        chunk_index = int(md.get("chunkIndex") or 0)
    except (TypeError, ValueError):
        chunk_index = 0
    try:
        total_chunks = int(md.get("totalChunks") or 1)
    except (TypeError, ValueError):
        total_chunks = 1
    raw_size = md.get("sizeBytes") or md.get("size") or md.get("fileSize")
    try:
        size_bytes = int(raw_size) if raw_size not in (None, "", 0) else None
    except (TypeError, ValueError):
        size_bytes = None
    mime_type = _infer_mime_type(
        modality, url, md.get("mimeType") or md.get("mime")
    )
    return {
        "type": "resource",
        "event_id": uuid.uuid4().hex,
        "url": url,
        "name": name,
        "display_name": _clean_filename(name),
        "file_id": file_id,
        "modality": modality,
        "mime_type": mime_type,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "size_bytes": size_bytes,
        "similarity": round(float(hit.get("similarity", 0.0) or 0.0), 4),
        "source": source_tag,
        "iter": iter_no,
    }


def _hits_to_resources(hits: list[dict], *, source_tag: str, iter_no: int | None) -> list[dict]:
    return [r for r in (_hit_to_resource(h, source_tag=source_tag, iter_no=iter_no) for h in hits) if r]


def _resource_to_markdown(res: dict) -> str:
    """把一条 resource 渲染成 markdown 片段。"""
    url = res.get("url") or ""
    name = res.get("name") or "未命名"
    modality = (res.get("modality") or "file").lower()
    if modality == "image":
        return f"![{name}]({url})"
    if modality in ("audio", "video"):
        return f"[{name}]({url})"
    return f"[{name}]({url})"


def _resources_as_appendix(
    resources: list[dict], existing_text: str
) -> str:
    """把资源列表格式化为可追加到 final_text 末尾的 markdown 段落。

    跳过 ``existing_text`` 已出现的 URL（避免 LLM 主动写了又被追加一遍）。
    返回空串表示无须追加。
    """
    seen: set[str] = set()
    out: list[str] = []
    for res in resources:
        url = res.get("url") or ""
        if not url or url in seen:
            continue
        if url in existing_text:
            continue
        seen.add(url)
        out.append(_resource_to_markdown(res))
    if not out:
        return ""
    return "\n\n附件：\n" + "\n".join(out)


# ---------- 工具结果 → LLM 可见的自然语言 ----------

def _tool_result_to_text(name: str, result: dict) -> str:
    if not result.get("ok"):
        return f"[tool:{name}] failed: {result.get('error')}"
    data = result.get("data") or {}
    if name == "understand_image":
        desc = data.get("description") or ""
        return f"[image] {desc}".strip()
    if name == "understand_audio":
        text = data.get("text") or ""
        return f"[audio transcript] {text}".strip()
    if name == "search_memory":
        hits = data.get("hits") or []
        causal = data.get("causal") or []
        modality_counts = data.get("modality_counts") or {}
        head_parts: list[str] = []
        for h in hits[:5]:
            sim = h.get("similarity", 0)
            modality = h.get("modality") or "memory"
            content = (h.get("content") or "")[:80]
            md = h.get("metadata") or {}
            # 对图像命中额外暴露 sourceUrl/fileName，方便 LLM 决定是否再调 understand_image
            extra = ""
            if modality == "image" and md.get("sourceUrl"):
                extra = f" url={md['sourceUrl']}"
            elif md.get("fileName"):
                extra = f" file={md['fileName']}"
            head_parts.append(f"[{modality}]({sim:.2f}) {content}{extra}")
        head = " | ".join(head_parts)
        cause_text = " ; ".join(
            f"[{c.get('emotion_tag','?')}] {c.get('content','')[:80]}" for c in causal[:3]
        )
        summary = f"[memory hits] {head}"
        if modality_counts:
            summary += f"\n[modality] {modality_counts}"
        if cause_text:
            summary += f"\n[causal] {cause_text}"
        return summary.strip()
    if name == "analyze_emotion":
        emo = data.get("emotion", "neutral")
        intensity = data.get("intensity", 0.0)
        reason = data.get("reason", "")
        return f"[emotion] {emo} (intensity={intensity:.2f}) {reason}".strip()
    return f"[tool:{name}] {json.dumps(data, ensure_ascii=False)[:300]}"


# ---------- ReAct 主循环 ----------

async def _react_loop(
    user_id: str,
    session_id: str,
    messages: list[dict[str, str]],
    max_iter: int,
) -> tuple[str, list[dict[str, Any]]]:
    """执行 ReAct 循环：每次让 LLM 决定是否调用工具，循环直到回复或耗尽迭代。

    返回 (final_assistant_text, tool_call_log)
    """
    client = get_llm_client()
    tool_log: list[dict[str, Any]] = []
    final_text = ""
    decided_final = False
    exhausted = False

    for i in range(max(1, max_iter)):
        # 让 LLM 判断是否需要调用工具
        decision_messages = list(messages) + [
            {
                "role": "system",
                "content": (
                    "如果你需要调用工具，**只**输出严格的 JSON："
                    '{"tool": "<name>", "args": {...}}。'
                    "如果你认为信息已经足够，输出自然语言回复。"
                ),
            }
        ]
        t0 = time.perf_counter()
        raw_text = ""
        try:
            resp = await client.chat(decision_messages, max_tokens=400, temperature=0.3)
            raw_text = resp.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            # 剥离思考块，避免影响 tool_call 解析
            text = _strip_think(raw_text)
        except Exception as e:
            logger.warning(
                "react decision failed",
                extra=merge_extra(stage="react_decision", event="error", iter=i, error=str(e)[:200]),
            )
            text = ""
        decision_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "react decision ok" if text else "react decision empty",
            extra=merge_extra(
                stage="react_decision",
                event="ok" if text else "empty",
                iter=i,
                out_len=len(text),
                has_think="<think>" in raw_text,
                duration_ms=decision_ms,
            ),
        )

        call = parse_tool_call(text)
        if not call:
            final_text = text
            decided_final = True
            logger.info(
                "react final (no tool call)",
                extra=merge_extra(stage="react_loop", event="final", iter=i, out_len=len(text)),
            )
            break

        name = call.get("tool")
        args = call.get("args") or {}
        # 把 user_id / session_id 注入到工具入参（不影响其他参数）
        if isinstance(args, dict):
            args.setdefault("user_id", user_id)
            args.setdefault("session_id", session_id)

        logger.info(
            "react iter dispatch",
            extra=merge_extra(
                stage="react_loop",
                event="dispatch",
                iter=i,
                tool=name,
                args_summary=json.dumps(args, ensure_ascii=False)[:200],
            ),
        )
        t1 = time.perf_counter()
        try:
            result = await dispatch_tool(name, **args)
        except Exception as e:
            tool_ms = round((time.perf_counter() - t1) * 1000, 2)
            logger.exception(
                "react tool dispatch failed",
                extra=merge_extra(
                    stage="react_loop",
                    event="tool_error",
                    iter=i,
                    tool=name,
                    duration_ms=tool_ms,
                    error=str(e)[:300],
                ),
            )
            break
        tool_ms = round((time.perf_counter() - t1) * 1000, 2)
        logger.info(
            "react tool result",
            extra=merge_extra(
                stage="react_loop",
                event="tool_ok",
                iter=i,
                tool=name,
                ok=bool(result.ok),
                err=(result.error or "")[:160] if not result.ok else "",
                duration_ms=tool_ms,
            ),
        )
        tool_log.append({"iter": i, "tool": name, "args": args, "result": result.to_dict()})

        # 把工具结果追加进 messages，让 LLM 下一轮看到
        result_text = _tool_result_to_text(name, result.to_dict())
        messages.append({"role": "assistant", "content": f"[calling tool: {name}]"})
        messages.append({"role": "user", "content": result_text})
    else:
        exhausted = True

    if exhausted and not decided_final:
        # 迭代耗尽也没拿到自然语言回复：让 LLM 直接总结
        t2 = time.perf_counter()
        try:
            resp = await client.chat(messages, max_tokens=600, temperature=0.4)
            final_text = resp.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            logger.info(
                "react exhausted → summary",
                extra=merge_extra(
                    stage="react_loop",
                    event="summary",
                    iter=max_iter - 1,
                    out_len=len(final_text),
                    duration_ms=round((time.perf_counter() - t2) * 1000, 2),
                ),
            )
        except Exception as e:
            logger.exception(
                "react summary failed",
                extra=merge_extra(
                    stage="react_loop",
                    event="summary_error",
                    error=str(e)[:200],
                ),
            )
            final_text = ""
        logger.info(
            "react exhausted",
            extra=merge_extra(stage="react_loop", event="exhausted", max_iter=max_iter),
        )

    final_text = _strip_think(final_text)
    logger.info(
        "react loop end",
        extra=merge_extra(
            stage="react_loop",
            event="end",
            tools_called=len(tool_log),
            exhausted=exhausted,
            final_len=len(final_text),
        ),
    )
    return final_text, tool_log


# ---------- 公开接口 ----------

async def chat_stream(
    user_id: str,
    session_id: str,
    user_msg: str,
) -> AsyncIterator[dict[str, Any]]:
    """流式 chat：预注入 → ReAct → 流式级联输出。

    yield 事件类型：
    - {"type": "context", "persona": ..., "l0_count": int, "l1_count": int}
    - {"type": "tool", "name": str, "iter": int, "ok": bool, "summary": str}
    - {"type": "prefix", "text": str}
    - {"type": "delta", "text": str}
    - {"type": "done", "full": str}
    """
    # 注：记忆抽取不在此处触发（避免在事件循环里 fire-and-forget 后立刻返回，
    # 导致下一次 /chat 来时记忆尚未落库）。由调用方根据 stream 模式决定 await 或 fire-and-forget。
    settings = get_settings().memory

    msg_len = len(user_msg or "")
    logger.info(
        "chat_stream start",
        extra=merge_extra(
            stage="chat_stream",
            event="start",
            msg_len=msg_len,
            msg_preview=(user_msg or "")[:120],
            react_max_iter=settings.react_max_iter,
        ),
    )

    # 0) 意图识别：决定是否触发 RAG 跨模态检索 / 是否注入 L1 hint
    intent_res = await classify_intent(user_msg)
    intent = intent_res.intent
    logger.info(
        "react decision",
        extra=merge_extra(
            stage="chat_stream",
            event="react_decision",
            intent=intent.value,
            intent_source=intent_res.source,
            intent_ms=intent_res.duration_ms,
        ),
    )

    # 1) 预注入上下文（仅 image_search 触发 multimodal_search，其余走轻量路径）
    t_ctx = time.perf_counter()
    with log_stage(logger, "context_build", start_msg="building chat context") as ctx_meta:
        ctx = await build_chat_context(
            user_id,
            user_msg,
            enable_multimodal=(intent == Intent.IMAGE_SEARCH),
        )
        l0_list = ctx.get("l0_memories", []) or []
        ctx_meta.update(
            persona_len=len(ctx.get("persona", "") or ""),
            l0_count=len(l0_list),
            l1_summary_count=len(ctx.get("recent_summaries", []) or []),
            l1_hit_count=len(ctx.get("l1_hits", []) or []),
            l0_preview=" | ".join(l0_list[:2])[:200],
            intent=intent.value,
        )
    yield {
        "type": "context",
        "intent": intent.value,
        "intent_source": intent_res.source,
        "intent_ms": intent_res.duration_ms,
        "persona_len": len(ctx["persona"] or ""),
        "l0_count": len(ctx["l0_memories"]),
        "l1_count": len(ctx["recent_summaries"]),
    }
    # 跟踪本次请求里所有可作为附件的资源，最后拼到 final_text 末尾
    emitted_resources: list[dict] = []
    # 把 L1 预注入里可作为附件展示的命中（图片/音频/文件）作为 resource 事件推给前端
    for res in _hits_to_resources(
        ctx.get("l1_hits", []) or [], source_tag="l1_hint", iter_no=None
    ):
        emitted_resources.append(res)
        yield res

    system_prompt = build_system_prompt(
        persona=ctx["persona"],
        l0_memories=ctx["l0_memories"],
        recent_summaries=ctx["recent_summaries"],
    )

    final_text = ""

    # 始终走 ReAct 循环；仅在非 chat 意图下注入 L1 hint（chat 类无 RAG 上下文，
    # 注入反而误导 LLM）。
    l1_hint = "\n".join(
        f"- ({h.get('similarity', 0):.2f}) {h.get('content', '')[:200]}"
        for h in ctx["l1_hits"][:5]
    )
    base_seed: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]
    if intent != Intent.CHAT:
        base_seed.append(
            {"role": "system", "content": f"【L1 检索预注入】\n{l1_hint or '（无）'}"}
        )
    base_seed.append({"role": "user", "content": user_msg})

    logger.info(
        "react seed ready",
        extra=merge_extra(
            stage="chat_stream",
            event="seed_ready",
            seed_len=len(base_seed),
            system_len=sum(len(m["content"]) for m in base_seed if m["role"] == "system"),
        ),
    )

    final_text, tool_log = await _react_loop(
        user_id=user_id,
        session_id=session_id,
        messages=base_seed,
        max_iter=settings.react_max_iter,
    )
    for entry in tool_log:
        yield {
            "type": "tool",
            "name": entry["tool"],
            "iter": entry["iter"],
            "ok": bool(entry["result"].get("ok")),
            "summary": _tool_result_to_text(entry["tool"], entry["result"])[:200],
        }
        # 把工具结果里可下载的资源（图片/音频/文件 URL）作为 resource 事件推给前端
        tool_data = (entry.get("result") or {}).get("data") or {}
        tool_hits = tool_data.get("hits") or []
        for res in _hits_to_resources(
            tool_hits, source_tag=entry["tool"], iter_no=entry["iter"]
        ):
            emitted_resources.append(res)
            yield res
    if final_text:
        # LLM 经常在裸 URL 前自作主张加 https://，在 yield 前清掉
        final_text = _strip_url_schemes(final_text)
        # 直接输出 final_text（不走级联以保证一致性）
        yield {"type": "prefix", "text": ""}
        yield {"type": "delta", "text": final_text}
    # 把可下载资源以 markdown 形式追加到 final_text 末尾，
    # 让只读 full 文本的前端也能拿到 URL（同时跳过 LLM 已经写过的）。
    appendix = _resources_as_appendix(emitted_resources, final_text or "")
    if appendix:
        final_text = (final_text or "") + appendix
        yield {"type": "delta", "text": appendix}
    logger.info(
        "chat_stream end",
        extra=merge_extra(
            stage="chat_stream",
            event="end",
            final_len=len(final_text),
            tool_count=len(tool_log),
            context_build_ms=round((time.perf_counter() - t_ctx) * 1000, 2),
        ),
    )
    yield {"type": "done", "full": final_text}


async def chat_collect(user_id: str, session_id: str, user_msg: str) -> dict:
    """一次性收集 chat 结果：等流结束 → 同步等待记忆抽取落库 → 再返回。

    与 chat_stream 不同：chat_collect 在生成完整回复后立即 await 记忆抽取，
    保证下一次 /chat 调用进来时记忆已经落库。
    """
    events: list[dict[str, Any]] = []
    full = ""
    started = time.time()
    async for ev in chat_stream(user_id, session_id, user_msg):
        events.append(ev)
        if ev["type"] == "done":
            full = ev["full"]
    # 同步路径：必须等记忆抽取落库，否则下一次对话拿不到刚产生的记忆
    settings = get_settings().memory
    if settings.enable_async_extract and full:
        t = time.perf_counter()
        try:
            await extract_and_archive_async(
                user_id=user_id,
                session_id=session_id,
                user_msg=user_msg,
                assistant_msg=full,
            )
            logger.info(
                "memory extract (sync) ok",
                extra=merge_extra(
                    stage="memory_extract_sync",
                    event="ok",
                    duration_ms=round((time.perf_counter() - t) * 1000, 2),
                ),
            )
            events.append({"type": "memory_extracted", "ok": True})
        except Exception as e:
            logger.exception(
                "memory extract (sync) failed",
                extra=merge_extra(
                    stage="memory_extract_sync",
                    event="error",
                    duration_ms=round((time.perf_counter() - t) * 1000, 2),
                    error=str(e)[:300],
                ),
            )
            events.append({"type": "memory_extracted", "ok": False, "error": str(e)[:200]})
    return {
        "events": events,
        "full": full,
        "latency_ms": int((time.time() - started) * 1000),
    }