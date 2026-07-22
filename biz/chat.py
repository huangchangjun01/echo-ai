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
from utils.request_context import log_exception, log_silent_failure, log_stage, merge_extra

logger = logging.getLogger(__name__)


# ---------- 意图 → 工具偏好提示 ----------
#
# 设计意图：每次 chat 入口的「意图分类」已经识别出用户想要哪一种能力，下面这张表把
# 意图翻译成一段「system 级 hint」，与 L1 hint 一起注入到 ReAct seed，让 LLM 在决策
# 时知道「优先用哪一把工具」，避免每次都从 4 把工具里自由试错。
#
# 注意：本提示不是「hard dispatch」——LLM 仍拥有 ReAct 主控权；判断「不需要任何工具」
# 也属于合法决策。这里只是把意图的偏好提前告知 LLM，让它在第一次决策时就能命中。
#
# 同 l1_hint：chat 意图不注入工具提示（LLM 默认就该走纯聊天路径，注入反而干扰）。
_INTENT_TOOL_HINT: dict[Intent, str] = {
    Intent.RECALL: (
        "【意图偏好】用户希望『回忆 / 翻历史』。"
        "请优先调用 search_memory 检索 EchoMemory + EchoDoc 中保存的相关记忆，"
        "拿到命中后再用自然语言整理。不要调用 understand_image / understand_audio。"
    ),
    Intent.TEXT_SEARCH: (
        "【意图偏好】用户希望『按文搜文』。"
        "请优先调用 search_memory（文本向量检索路径），命中后整合成回答；"
        "若命中附带可下载的图片/视频 URL，可视情况再调用 understand_image 复述内容。"
    ),
    Intent.DOC_SEARCH: (
        "【意图偏好】用户希望『按主题找文件 / 合同 / 资料』。"
        "请优先调用 search_memory（重点走 EchoDoc 文档集合）；"
        "若需要再次确认图片/截图内容，再调用 understand_image。"
    ),
    Intent.IMAGE_SEARCH: (
        "【意图偏好】用户希望『找一张图 / 看一张图 / 上传图片分析』。"
        "若本轮附带图片数据，优先调用 understand_image 看图说话；"
        "若是『回忆某张历史图』（如『上次那只猫』），先调 search_memory 找出命中，"
        "再视情况对命中的图片 URL 调 understand_image 复述。"
    ),
    Intent.CHAT: (
        "【意图偏好】本轮是纯聊天 / 寒暄。"
        "如无强需求，直接自然语言回复即可，不必调用工具。"
    ),
}


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
    """规范化 sourceUrl：剥离多余的 scheme 前缀后，按需补回 ``https://``。

    上下文：data 里的 sourceUrl 形态多样：
    - ``https://cdn.example.com/foo.jpg`` / ``http://...`` → 保留 https
    - ``//cdn.example.com/foo.jpg`` → 补 https
    - ``cdn.example.com/foo.jpg``（裸域名，Qiniu 存储常用形态）→ 补 https
    前端可直接用返回值做 ``<img src>`` / 下载，不再需要自行拼接 scheme。
    """
    if not url:
        return ""
    s = url.strip()
    if not s:
        return ""
    for prefix in ("https://", "http://"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break
    if s.startswith("//"):
        s = s[2:]
    return f"https://{s}"


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

async def _resolve_tools(
    user_id: str,
    session_id: str,
    messages: list[dict[str, str]],
    max_iter: int,
    role_id: str = "default",
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """ReAct 工具解析：逐轮让 LLM 决定是否调用工具并执行，直到 LLM 不再要求工具或迭代耗尽。

    **不生成最终自然语言回复**——最终回复由 chat_stream 通过 cascade 流式产出。

    返回 (tool_log, tool_turns)。tool_turns 是可追加进「最终生成上下文」的对话轮次
    （形如 assistant「[calling tool: x]」+ user「工具结果」），让最终生成看到工具产出。
    """
    client = get_llm_client()
    tool_log: list[dict[str, Any]] = []
    tool_turns: list[dict[str, str]] = []
    exhausted = False

    for i in range(max(1, max_iter)):
        # 让 LLM 判断是否需要调用工具；不需要时随口回一句即可（真正回答稍后单独生成）
        decision_messages = list(messages) + tool_turns + [
            {
                "role": "system",
                "content": (
                    "如果你需要调用工具，**只**输出严格的 JSON："
                    '{"tool": "<name>", "args": {...}}。'
                    '如果不需要任何工具，只回复 "OK" 即可（真正的回答会另行生成）。'
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
            log_exception(
                logger,
                "react decision failed",
                exc=e,
                level=logging.WARNING,
                stage="react_decision",
                event="error",
                iter=i,
                msg_count=len(messages),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
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
            logger.info(
                "react no tool call (hand off to cascade)",
                extra=merge_extra(stage="react_loop", event="no_tool", iter=i, out_len=len(text)),
            )
            break

        name = call.get("tool")
        args = call.get("args") or {}
        # 把 user_id / session_id 注入到工具入参（不影响其他参数）
        if isinstance(args, dict):
            args.setdefault("user_id", user_id)
            args.setdefault("session_id", session_id)
            args.setdefault("role_id", role_id)

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
            log_exception(
                logger,
                "react tool dispatch failed",
                exc=e,
                stage="react_loop",
                event="tool_error",
                iter=i,
                tool=name,
                args_summary=json.dumps(args, ensure_ascii=False)[:200],
                duration_ms=tool_ms,
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

        # 把工具结果追加进 tool_turns，让 LLM 下一轮决策 + 最终生成都能看到
        result_text = _tool_result_to_text(name, result.to_dict())
        tool_turns.append({"role": "assistant", "content": f"[calling tool: {name}]"})
        tool_turns.append({"role": "user", "content": result_text})
    else:
        exhausted = True

    logger.info(
        "react loop end",
        extra=merge_extra(
            stage="react_loop",
            event="end",
            tools_called=len(tool_log),
            exhausted=exhausted,
        ),
    )
    return tool_log, tool_turns


# ---------- 公开接口 ----------

async def chat_stream(
    user_id: str,
    session_id: str,
    user_msg: str,
    role_id: str = "default",
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
            role_id=role_id,
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

    # 工具决策 seed：仅在非 chat 意图下注入 L1 hint + 工具偏好（chat 类无 RAG 上下文，
    # 注入反而误导 LLM）。这份 seed 只用于「该不该调工具、调哪把」的 ReAct 决策。
    l1_hint = "\n".join(
        f"- ({h.get('similarity', 0):.2f}) {h.get('content', '')[:200]}"
        for h in ctx["l1_hits"][:5]
    )
    tool_seed: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]
    if intent != Intent.CHAT:
        tool_seed.append(
            {"role": "system", "content": f"【L1 检索预注入】\n{l1_hint or '（无）'}"}
        )
        tool_hint = _INTENT_TOOL_HINT.get(intent)
        if tool_hint:
            tool_seed.append({"role": "system", "content": tool_hint})
    tool_seed.append({"role": "user", "content": user_msg})

    logger.info(
        "react seed ready",
        extra=merge_extra(
            stage="chat_stream",
            event="seed_ready",
            intent=intent.value,
            intent_source=intent_res.source,
            seed_len=len(tool_seed),
            system_len=sum(len(m["content"]) for m in tool_seed if m["role"] == "system"),
            injected_l1_hint=intent != Intent.CHAT,
            injected_tool_hint=bool(intent != Intent.CHAT and _INTENT_TOOL_HINT.get(intent)),
        ),
    )

    # ReAct 总是跑：即便意图是 CHAT，也给 LLM 一次机会决定要不要调工具。
    # LLM 不调 → 走纯聊天快速路径；调了 → 拿到 tool_turns 进入最终生成。
    # （规则快速路径识别出的寒暄已通过 tool_seed 的 system 提示引导，LLM 几乎不会误调。）
    tool_log: list[dict[str, Any]] = []
    tool_turns: list[dict[str, str]] = []
    tool_log, tool_turns = await _resolve_tools(
        user_id=user_id,
        session_id=session_id,
        messages=tool_seed,
        max_iter=settings.react_max_iter,
        role_id=role_id,
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

    # 「最终回复生成」上下文：干净的对话（不带工具偏好/JSON 框架），含工具结果 + 收尾指令。
    # 交由 cascade 做「小模型前缀 + 大模型实时续写」的流式生成。
    final_messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]
    if intent != Intent.CHAT:
        final_messages.append(
            {"role": "system", "content": f"【L1 检索预注入】\n{l1_hint or '（无）'}"}
        )
    final_messages.append({"role": "user", "content": user_msg})
    final_messages += tool_turns
    if tool_turns:
        final_messages.append(
            {
                "role": "system",
                "content": "基于以上工具返回的信息，用自然语言直接回复用户；不要再调用工具或输出 JSON。",
            }
        )

    # 流式级联：小模型前缀（首屏）+ 大模型逐 chunk 实时续写。
    # LLM 常在裸 URL 前自作主张加 https://，逐段清掉 scheme 前缀后再下发。
    async for ev in cascade_chat(
        final_messages, prefix_tokens=settings.cascade_prefix_tokens
    ):
        et = ev.get("type")
        if et == "prefix":
            yield {"type": "prefix", "text": _strip_url_schemes(ev.get("text") or "")}
        elif et == "delta":
            yield {"type": "delta", "text": _strip_url_schemes(ev.get("text") or "")}
        elif et == "done":
            final_text = _strip_url_schemes(ev.get("full") or "")

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


async def chat_collect(user_id: str, session_id: str, user_msg: str, role_id: str = "default") -> dict:
    """一次性收集 chat 结果：等流结束 → 同步等待记忆抽取落库 → 再返回。

    与 chat_stream 不同：chat_collect 在生成完整回复后立即 await 记忆抽取，
    保证下一次 /chat 调用进来时记忆已经落库。
    """
    events: list[dict[str, Any]] = []
    full = ""
    started = time.time()
    async for ev in chat_stream(user_id, session_id, user_msg, role_id):
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
                role_id=role_id,
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
            log_exception(
                logger,
                "memory extract (sync) failed",
                exc=e,
                level=logging.ERROR,
                stage="memory_extract_sync",
                event="error",
                user_id=user_id,
                session_id=session_id,
                user_msg_len=len(user_msg or ""),
                assistant_msg_len=len(full or ""),
                duration_ms=round((time.perf_counter() - t) * 1000, 2),
            )
            events.append({"type": "memory_extracted", "ok": False, "error": str(e)[:200]})
    return {
        "events": events,
        "full": full,
        "latency_ms": int((time.time() - started) * 1000),
    }