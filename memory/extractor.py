"""记忆抽取：从一段对话中提取原子事实 / 因果 / 情感 / 关系，分层归档。

流程（按用户要求）：
1. LLM 抽取（原子事实 + 因果关系）
2. 向量化（BGE-M3 → weaviate）
3. 语义去重（weaviate 向量 + LLM 判重）
4. 关系识别（causes/update/contradict/extend）
5. 情感标注（emotion_tag + intensity）
6. 分层归档（L0/L1/L2 → MySQL）
7. 相似记忆合并 + 矛盾清理
8. 记忆摘要
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any

from config.config import get_settings
from llm.think import _strip_think
from utils.request_context import log_exception, log_silent_failure, log_stage, merge_extra

logger = logging.getLogger(__name__)


# ---------- JSON 修复兜底 ----------

# 字符串值内出现裸 " 是 LLM 生成 JSON 时最常见的坏 JSON 形态。Prompt 已要求用中文引号，
# 但模型仍偶发不遵守；这里用状态机扫描 + look-ahead 判定"闭合引号 vs 嵌入引号"做一次就地修复，
# 让单条坏 JSON 不至于让整段对话的记忆全部丢失。
_STRING_END_LOOKAHEAD = set(",}]:")
_WS = set(" \t\r\n")


def _repair_json(text: str) -> str:
    """对 LLM 输出的 JSON 做就地修复。

    规则：按字符扫描；进入字符串后，遇到 " 时向后跳过空白，若下一个非空白字符是
    , } ] : 或 EOF 则视为闭合引号原样保留；否则视为嵌入引号转义为 \\"。
    转义反斜杠与已存在的合法转义序列保持原样。
    """
    out: list[str] = []
    in_string = False
    escape_next = False
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if escape_next:
            out.append(ch)
            escape_next = False
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            escape_next = True
            i += 1
            continue
        if ch == '"':
            if not in_string:
                out.append(ch)
                in_string = True
                i += 1
                continue
            j = i + 1
            while j < n and text[j] in _WS:
                j += 1
            if j >= n or text[j] in _STRING_END_LOOKAHEAD:
                out.append(ch)
                in_string = False
            else:
                out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


# ---------- LLM 抽取 ----------

def _normalize_fact_item(it: Any) -> dict | None:
    """把 LLM 输出的单条记忆规整成统一结构；非法条目返回 None。"""
    if not isinstance(it, dict):
        return None
    fact = (it.get("fact") or "").strip()
    if not fact:
        return None
    return {
        "fact": fact,
        "causes": (it.get("causes") or "").strip(),
        "level": (it.get("level") or "L1").upper(),
        "emotion": (it.get("emotion") or "neutral").lower(),
        "intensity": float(it.get("intensity", 0.0) or 0.0),
        "relation": (it.get("relation") or "").lower(),
    }


def _parse_extracted_items(content: str, *, stage: str, log_extra: dict, t0: float) -> list[dict]:
    """从 LLM 原始输出里解析出记忆数组（含 <think> 剥离 + 坏 JSON 修复兜底）。"""
    cleaned = _strip_think(content or "")
    m = re.search(r"\[[\s\S]*?\]", cleaned)
    if not m:
        logger.info(
            f"{stage} empty list",
            extra=merge_extra(
                stage=stage,
                event="empty",
                raw_preview=(content or "")[:120],
                cleaned_preview=(cleaned or "")[:120],
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                **log_extra,
            ),
        )
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        # 解析失败：先尝试状态机式修复（裸双引号场景），修复后再解析一次。
        repaired = _repair_json(m.group(0))
        try:
            items = json.loads(repaired)
            logger.info(
                f"{stage} json repaired",
                extra=merge_extra(
                    stage=stage,
                    event="repaired",
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    **log_extra,
                ),
            )
        except json.JSONDecodeError as e2:
            log_exception(
                logger,
                f"{stage} JSON parse failed",
                exc=e2,
                level=logging.WARNING,
                stage=stage,
                event="parse_error",
                raw_preview=(content or "")[:200],
                cleaned_preview=(cleaned or "")[:200],
                matched_preview=m.group(0)[:200],
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                **log_extra,
            )
            return []
    if not isinstance(items, list):
        return []
    out = [x for x in (_normalize_fact_item(it) for it in items) if x]
    logger.info(
        f"{stage} ok",
        extra=merge_extra(
            stage=stage,
            event="ok",
            count=len(out),
            raw_items=len(items),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            **log_extra,
        ),
    )
    return out


async def _llm_extract(user_msg: str, assistant_msg: str) -> list[dict]:
    from llm.client import get_llm_client
    from config.prompts import MEMORY_EXTRACT_SYSTEM, MEMORY_EXTRACT_USER_TEMPLATE

    client = get_llm_client()
    prompt_user = MEMORY_EXTRACT_USER_TEMPLATE.format(user_msg=user_msg, assistant_msg=assistant_msg)
    t0 = time.perf_counter()
    log_extra = {
        "user_msg_len": len(user_msg or ""),
        "assistant_msg_len": len(assistant_msg or ""),
    }
    try:
        resp = await client.chat(
            [
                {"role": "system", "content": MEMORY_EXTRACT_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            max_tokens=600,
            temperature=0.2,
        )
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        return _parse_extracted_items(content, stage="llm_extract", log_extra=log_extra, t0=t0)
    except Exception as e:
        log_exception(
            logger,
            "LLM extract failed",
            exc=e,
            level=logging.WARNING,
            stage="llm_extract",
            event="error",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            **log_extra,
        )
        return []


async def _llm_extract_file(
    file_name: str, modality: str, desc: str, parsed_content: str
) -> list[dict]:
    """从文件描述 + 解析内容抽取记忆（复用对话抽取的解析/修复逻辑）。"""
    from llm.client import get_llm_client
    from config.prompts import FILE_MEMORY_EXTRACT_SYSTEM, FILE_MEMORY_EXTRACT_USER_TEMPLATE

    client = get_llm_client()
    prompt_user = FILE_MEMORY_EXTRACT_USER_TEMPLATE.format(
        file_name=file_name or "（未命名）",
        modality=modality or "unknown",
        desc=desc or "（无）",
        parsed_content=(parsed_content or "（无）")[:4000],
    )
    t0 = time.perf_counter()
    log_extra = {
        "file_name": file_name,
        "modality": modality,
        "desc_len": len(desc or ""),
        "content_len": len(parsed_content or ""),
    }
    try:
        resp = await client.chat(
            [
                {"role": "system", "content": FILE_MEMORY_EXTRACT_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            max_tokens=600,
            temperature=0.2,
        )
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        return _parse_extracted_items(content, stage="llm_extract_file", log_extra=log_extra, t0=t0)
    except Exception as e:
        log_exception(
            logger,
            "LLM extract (file) failed",
            exc=e,
            level=logging.WARNING,
            stage="llm_extract_file",
            event="error",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            **log_extra,
        )
        return []


# ---------- 向量化 + weaviate 写入 ----------

async def _vectorize_and_store(
    user_id: str,
    facts: list[str],
    base_metadata: dict,
    role_id: str = "default",
) -> list[str]:
    """返回每条 fact 对应的 weaviate uuid。失败返回空 list。"""
    if not facts:
        return []
    t0 = time.perf_counter()
    try:
        from embedding.bge_m3 import embed_texts
        from vector.memory_store import get_memory_vector_store

        vecs_t = time.perf_counter()
        vectors = await asyncio.to_thread(embed_texts, facts)
        embed_ms = round((time.perf_counter() - vecs_t) * 1000, 2)

        vs = await asyncio.to_thread(get_memory_vector_store)
        ids = [f"mem-{uuid.uuid4().hex}" for _ in facts]
        metadatas = [
            {
                **base_metadata,
                "userId": user_id,
                "roleId": role_id,
                "chunkIndex": i,
                "factIndex": i,
            }
            for i in range(len(facts))
        ]
        write_t = time.perf_counter()
        await asyncio.to_thread(
            vs.add_texts,
            ids,
            facts,
            metadatas,
            vectors,
        )
        write_ms = round((time.perf_counter() - write_t) * 1000, 2)
        logger.info(
            "vectorize_and_store ok",
            extra=merge_extra(
                stage="vectorize_and_store",
                event="ok",
                user_id=user_id,
                count=len(facts),
                embed_ms=embed_ms,
                write_ms=write_ms,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return ids
    except Exception as e:
        log_exception(
            logger,
            "vectorize_and_store failed",
            exc=e,
            level=logging.WARNING,
            stage="vectorize_and_store",
            event="error",
            user_id=user_id,
            count=len(facts),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return []


# ---------- 去重判重 ----------

async def _dedup_one(user_id: str, fact: str, vector: list[float], threshold: float, role_id: str = "default") -> dict:
    """对单条 fact 与已有记忆做向量检索 + LLM 判重。返回 {"duplicate": bool, ...}。"""
    settings = get_settings().memory
    t0 = time.perf_counter()
    try:
        from vector.memory_store import get_memory_vector_store

        vs = await asyncio.to_thread(get_memory_vector_store)

        def _fixed_embed(_texts: list[str]) -> list[list[float]]:
            return [vector]

        result = await asyncio.to_thread(
            vs.query,
            fact,
            5,
            _fixed_embed,
            {"userId": user_id, "roleId": role_id},
        )
        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        if not docs:
            return {"duplicate": False, "merged": fact, "relation": ""}
        # 取最高相似度
        top_sim = max((1.0 - float(d) for d in distances), default=0.0)
        if top_sim < threshold:
            return {"duplicate": False, "merged": fact, "relation": ""}
        # 让 LLM 判定
        try:
            from llm.client import get_llm_client
            from config.prompts import MEMORY_DEDUP_SYSTEM

            client = get_llm_client()
            resp = await client.chat(
                [
                    {"role": "system", "content": MEMORY_DEDUP_SYSTEM},
                    {
                        "role": "user",
                        "content": f"记忆A：{docs[0]}\n记忆B：{fact}",
                    },
                ],
                max_tokens=200,
                temperature=0.1,
            )
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            # 推理型模型会先吐 <think> 块；不剥掉会导致贪婪正则吞掉思考块后 JSON
            # 解析失败，与 _llm_extract 中的处理一致。
            cleaned = _strip_think(content or "")
            m = re.search(r"\{[\s\S]*?\}", cleaned)
            obj: dict | None = None
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    if isinstance(parsed, dict):
                        obj = parsed
                except json.JSONDecodeError as e:
                    # 兜底：尝试用状态机修复裸引号。
                    repaired = _repair_json(m.group(0))
                    try:
                        parsed = json.loads(repaired)
                        if isinstance(parsed, dict):
                            obj = parsed
                            logger.info(
                                "dedup json repaired",
                                extra=merge_extra(
                                    stage="dedup_one",
                                    event="repaired",
                                    user_id=user_id,
                                    top_sim=round(top_sim, 4),
                                ),
                            )
                    except json.JSONDecodeError as e2:
                        log_exception(
                            logger,
                            "dedup JSON parse failed, fallback heuristic",
                            exc=e2,
                            level=logging.WARNING,
                            stage="dedup_one",
                            event="parse_error",
                            user_id=user_id,
                            top_sim=round(top_sim, 4),
                            threshold=threshold,
                            raw_preview=(content or "")[:200],
                            cleaned_preview=(cleaned or "")[:200],
                        )
            if obj is not None:
                decision = {
                    "duplicate": bool(obj.get("duplicate", False)),
                    "merged": obj.get("merged", fact if not obj.get("duplicate") else docs[0]),
                    "relation": obj.get("relation", "") if not obj.get("duplicate") else "",
                }
                logger.info(
                    "dedup_one llm",
                    extra=merge_extra(
                        stage="dedup_one",
                        event="llm_decision",
                        user_id=user_id,
                        duplicate=decision["duplicate"],
                        relation=decision["relation"],
                        top_sim=round(top_sim, 4),
                        threshold=threshold,
                        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    ),
                )
                return decision
        except Exception as e:
            log_exception(
                logger,
                "LLM dedup failed, fallback heuristic",
                exc=e,
                level=logging.WARNING,
                stage="dedup_one",
                event="llm_error",
                user_id=user_id,
                top_sim=round(top_sim, 4) if 'top_sim' in locals() else None,
                threshold=threshold,
            )
        # 启发式：top_sim 高即视作重复
        decision = {
            "duplicate": top_sim >= threshold,
            "merged": fact,
            "relation": "extend",
        }
        logger.info(
            "dedup_one heuristic",
            extra=merge_extra(
                stage="dedup_one",
                event="heuristic",
                user_id=user_id,
                duplicate=decision["duplicate"],
                top_sim=round(top_sim, 4),
                threshold=threshold,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return decision
    except Exception as e:
        log_exception(
            logger,
            "dedup_one failed",
            exc=e,
            level=logging.WARNING,
            stage="dedup_one",
            event="error",
            user_id=user_id,
            fact_preview=fact[:80],
        )
        return {"duplicate": False, "merged": fact, "relation": ""}


# ---------- 记忆摘要 ----------

async def _summarize(user_id: str, contents: list[str]) -> str:
    if not contents:
        return ""
    t0 = time.perf_counter()
    try:
        from llm.client import get_llm_client
        from llm.think import _strip_think

        client = get_llm_client()
        prompt = (
            "请把以下若干条记忆用 1~2 句中文提炼为一条 L2 摘要，保留关键事实和情感：\n"
            + "\n".join(f"- {c}" for c in contents)
        )
        resp = await client.chat(
            [
                {
                    "role": "system",
                    "content": "你是一名记忆摘要助手。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        text = resp.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        text = _strip_think(text)
        logger.info(
            "summarize ok",
            extra=merge_extra(
                stage="summarize",
                event="ok",
                user_id=user_id,
                input_count=len(contents),
                out_len=len(text),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return text
    except Exception as e:
        log_exception(
            logger,
            "summarize failed",
            exc=e,
            level=logging.WARNING,
            stage="summarize",
            event="error",
            user_id=user_id,
            input_count=len(contents),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2) if 't0' in locals() else None,
        )
        return ""


# ---------- MySQL 归档 ----------

_VALID_LEVELS = {"L0", "L1", "L2"}


def _is_placeholder_parsed_content(text: str) -> bool:
    """判定 parsed_content 是否为「解析失败占位文本」（无任何真实信号）。

    与 parsers/audio_parser.py:parse_audio 与 biz/ingest.py:_ingest_audio 中的占位
    文案对齐：``[音频转录失败] {file_name}``。占位文本喂给 LLM 会触发幻觉（基于
    文件名编造内容），所以这里直接判 False。

    防御性扩展：未来若新增视频/图片解析失败的占位（如 ``[视频关键帧抽取失败]``），
    只要加上对应前缀即可。
    """
    if not text:
        return False
    s = text.strip()
    placeholders = (
        "[音频转录失败]",
        "[视频关键帧抽取失败]",
        "[图片描述失败]",
        "[文档解析失败]",
    )
    return any(s.startswith(p) for p in placeholders)

# memory_relations 存在多种历史 schema：新表用 relation/weight，旧表用 relation_type/confidence，
# 迁移后的表可能两套都在。缓存一次实际列名，据此拼 INSERT，避免盲试触发 1054/1364 错误。
_REL_COLUMNS: set[str] | None = None


async def _memory_relations_columns() -> set[str]:
    """探测并缓存 memory_relations 的实际列名（进程内单次）。"""
    global _REL_COLUMNS
    if _REL_COLUMNS is not None:
        return _REL_COLUMNS
    from database import fetch_all

    try:
        rows = await fetch_all("SHOW COLUMNS FROM memory_relations")
        _REL_COLUMNS = {str(r.get("Field")) for r in rows if r.get("Field")}
    except Exception as e:
        log_silent_failure(
            logger,
            "probe memory_relations columns failed; assume current schema",
            exc=e,
            stage="archive",
            event="rel_cols_probe_error",
        )
        _REL_COLUMNS = {"user_id", "source_id", "target_id", "relation", "weight"}
    return _REL_COLUMNS


async def _archive(
    user_id: str,
    items: list[dict],
    vector_ids: list[str | None],
    summary: str,
    role_id: str = "default",
) -> list[int]:
    """把抽取结果写回 MySQL。返回新插入的 memory id 列表。"""
    from database import execute, fetch_one

    inserted: list[int] = []
    # 1) 先写 summary（L2）独立行
    summary_id: int | None = None
    if summary:
        try:
            await execute(
                """
                INSERT INTO memories (user_id, role_id, level, content, summary, emotion_tag, emotion_intensity, importance)
                VALUES (%s, %s, 'L2', %s, %s, 'neutral', 0.0, 0.7)
                """,
                (user_id, role_id, summary, summary),
            )
            row = await fetch_one("SELECT LAST_INSERT_ID() AS id")
            summary_id = int(row["id"]) if row else None
            logger.info(
                "archive summary inserted",
                extra=merge_extra(
                    stage="archive",
                    event="summary_inserted",
                    user_id=user_id,
                    summary_id=summary_id,
                    summary_len=len(summary),
                ),
            )
        except Exception as e:
            log_exception(
                logger,
                "archive summary failed",
                exc=e,
                level=logging.WARNING,
                stage="archive",
                event="summary_error",
                user_id=user_id,
                summary_len=len(summary) if 'summary' in locals() else None,
            )

    # 2) 写每条 L0/L1
    for i, item in enumerate(items):
        level = item.get("level", "L1").upper()
        if level not in _VALID_LEVELS:
            level = "L1"
        content = item.get("fact", "")
        if not content:
            continue
        emotion = (item.get("emotion") or "neutral").lower()
        intensity = float(item.get("intensity") or 0.0)
        v_id = vector_ids[i] if i < len(vector_ids) else None
        try:
            await execute(
                """
                INSERT INTO memories (user_id, role_id, level, content, summary, emotion_tag,
                    emotion_intensity, vector_id, parent_id, importance,
                    memory_type, category, embedding_dim)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    role_id,
                    level,
                    content,
                    None,
                    emotion,
                    intensity,
                    v_id,
                    summary_id if level == "L1" else None,
                    0.7 if level == "L0" else 0.5,
                    "fact",
                    "general",
                    1024 if v_id else None,
                ),
            )
            row = await fetch_one("SELECT LAST_INSERT_ID() AS id")
            new_id = int(row["id"]) if row else None
            if new_id:
                inserted.append(new_id)
        except Exception as e:
            log_exception(
                logger,
                "archive item failed",
                exc=e,
                level=logging.WARNING,
                stage="archive",
                event="item_error",
                user_id=user_id,
                memory_level=level,
                item_index=i if 'i' in locals() else None,
            )

    # 3) 关系表（兼容旧表：用 relation_type / confidence）
    relations_inserted = 0
    for i, item in enumerate(items):
        rel = (item.get("relation") or "").lower()
        if rel not in {"causes", "update", "contradict", "extend"}:
            continue
        if i >= len(inserted):
            continue
        target_id = inserted[i]
        try:
            row = await fetch_one(
                """
                SELECT id FROM memories
                WHERE user_id=%s AND role_id=%s AND level=%s AND id<%s
                ORDER BY id DESC LIMIT 1
                """,
                (user_id, role_id, item.get("level", "L1"), target_id),
            )
            if row:
                source_id = int(row["id"])
                # 按实际存在的列拼 INSERT：新表 relation/weight，旧表 relation_type/confidence，
                # 两套并存时全部写入以满足各自的 NOT NULL 约束。
                cols = await _memory_relations_columns()
                fields = ["user_id", "source_id", "target_id"]
                values: list[Any] = [user_id, source_id, target_id]
                if "role_id" in cols:
                    fields.append("role_id")
                    values.append(role_id)
                if "relation" in cols:
                    fields.append("relation")
                    values.append(rel)
                if "relation_type" in cols:
                    fields.append("relation_type")
                    values.append(rel)
                if "weight" in cols:
                    fields.append("weight")
                    values.append(1.0)
                if "confidence" in cols:
                    fields.append("confidence")
                    values.append(1.0)
                placeholders = ", ".join(["%s"] * len(values))
                await execute(
                    f"INSERT INTO memory_relations ({', '.join(fields)}) "
                    f"VALUES ({placeholders})",
                    tuple(values),
                )
                relations_inserted += 1
        except Exception as e:
            log_exception(
                logger,
                "insert relation failed",
                exc=e,
                level=logging.WARNING,
                stage="archive",
                event="relation_error",
                user_id=user_id,
                relation=rel if 'rel' in locals() else None,
            )

    logger.info(
        "archive done",
        extra=merge_extra(
            stage="archive",
            event="end",
            user_id=user_id,
            inserted=len(inserted),
            relations=relations_inserted,
            summary_id=summary_id,
        ),
    )
    return inserted


# ---------- 公开 API ----------

async def extract_and_archive(
    user_id: str,
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    role_id: str = "default",
) -> dict:
    """从一段对话抽取记忆并归档。返回 {count, inserted_ids, summary}。"""
    if not user_id or not (user_msg or assistant_msg):
        return {"count": 0, "inserted_ids": [], "summary": ""}

    settings = get_settings().memory
    t0 = time.perf_counter()

    # 1) 抽取
    items = await _llm_extract(user_msg, assistant_msg)
    if not items:
        logger.info(
            "extract_and_archive skip (no items)",
            extra=merge_extra(
                stage="extract_and_archive",
                event="empty",
                user_id=user_id,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return {"count": 0, "inserted_ids": [], "summary": ""}

    # 2) 去重判重
    deduped_items: list[dict] = []
    deduped_vectors: list[str | None] = []
    skipped = 0
    from embedding.bge_m3 import embed_texts

    facts = [it["fact"] for it in items]
    emb_t = time.perf_counter()
    vectors = await asyncio.to_thread(embed_texts, facts)
    emb_ms = round((time.perf_counter() - emb_t) * 1000, 2)

    for i, it in enumerate(items):
        v = vectors[i] if i < len(vectors) else []
        decision = await _dedup_one(
            user_id,
            it["fact"],
            v,
            settings.dedup_threshold,
            role_id,
        )
        if decision.get("duplicate"):
            skipped += 1
            continue
        merged = decision.get("merged") or it["fact"]
        new_it = dict(it)
        new_it["fact"] = merged
        rel = decision.get("relation") or it.get("relation") or ""
        if rel:
            new_it["relation"] = rel
        deduped_items.append(new_it)
        deduped_vectors.append(None)
    logger.info(
        "extract dedup summary",
        extra=merge_extra(
            stage="extract_and_archive",
            event="dedup",
            user_id=user_id,
            extracted=len(items),
            kept=len(deduped_items),
            skipped=skipped,
            threshold=settings.dedup_threshold,
            embed_ms=emb_ms,
        ),
    )

    # 3) 向量化 + weaviate
    v_ids = await _vectorize_and_store(
        user_id,
        [it["fact"] for it in deduped_items],
        {"source": "memory_extract", "sessionId": session_id},
        role_id,
    )

    # 4) 摘要
    summary = await _summarize(user_id, [it["fact"] for it in deduped_items])

    # 5) MySQL 归档
    inserted = await _archive(user_id, deduped_items, v_ids, summary, role_id)
    logger.info(
        "extract_and_archive done",
        extra=merge_extra(
            stage="extract_and_archive",
            event="end",
            user_id=user_id,
            extracted=len(items),
            kept=len(deduped_items),
            inserted=len(inserted),
            vector_stored=len(v_ids),
            summary_len=len(summary),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        ),
    )
    return {
        "count": len(inserted),
        "inserted_ids": inserted,
        "summary": summary,
    }


async def extract_and_archive_async(
    user_id: str,
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    role_id: str = "default",
) -> None:
    """后台任务：异步执行抽取；落库抽取日志。"""
    from database import execute

    log_id: int | None = None
    t0 = time.perf_counter()
    logger.info(
        "extract_and_archive_async start",
        extra=merge_extra(
            stage="extract_and_archive_async",
            event="start",
            user_id=user_id,
            session_id=session_id,
            user_msg_len=len(user_msg or ""),
            assistant_msg_len=len(assistant_msg or ""),
        ),
    )
    try:
        await execute(
            """
            INSERT INTO memory_extract_logs (user_id, session_id, user_msg, assistant_msg, status)
            VALUES (%s, %s, %s, %s, 'pending')
            """,
            (
                user_id,
                session_id,
                (user_msg or "")[:1000],
                (assistant_msg or "")[:2000],
            ),
        )
        from database import fetch_one

        row = await fetch_one("SELECT LAST_INSERT_ID() AS id")
        log_id = int(row["id"]) if row else None
    except Exception as e:
        log_exception(
            logger,
            "create extract log failed",
            exc=e,
            level=logging.WARNING,
            stage="extract_and_archive_async",
            event="log_create_error",
            user_id=user_id,
            session_id=session_id,
        )

    try:
        result = await extract_and_archive(user_id, session_id, user_msg, assistant_msg, role_id)
        if log_id is not None:
            try:
                from database import execute as _exe

                await _exe(
                    """
                    UPDATE memory_extract_logs
                    SET status='success', finished_at=NOW()
                    WHERE id=%s
                    """,
                    (log_id,),
                )
            except Exception as e:
                log_silent_failure(
                    logger,
                    "update extract log to success failed",
                    exc=e,
                    stage="extract_and_archive_async",
                    event="status_update_error",
                    log_id=log_id,
                    target_status="success",
                )
        logger.info(
            "memory extract done",
            extra=merge_extra(
                stage="extract_and_archive_async",
                event="ok",
                user_id=user_id,
                inserted=result.get("count", 0),
                summary_len=len(result.get("summary", "") or ""),
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
    except Exception as e:
        log_exception(
            logger,
            "memory extract failed",
            exc=e,
            level=logging.ERROR,
            include_traceback=True,
            stage="extract_and_archive_async",
            event="error",
            user_id=user_id,
            session_id=session_id,
            user_msg_len=len(user_msg or ""),
            assistant_msg_len=len(assistant_msg or ""),
            log_id=log_id,
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        if log_id is not None:
            try:
                from database import execute as _exe

                await _exe(
                    """
                    UPDATE memory_extract_logs
                    SET status='failed', error=%s, finished_at=NOW()
                    WHERE id=%s
                    """,
                    (str(e)[:500], log_id),
                )
            except Exception as nested_e:
                log_silent_failure(
                    logger,
                    "update extract log to failed failed",
                    exc=nested_e,
                    stage="extract_and_archive_async",
                    event="status_update_error",
                    log_id=log_id,
                    target_status="failed",
                    root_error=str(e)[:200],
                )


async def extract_from_file(
    user_id: str,
    role_id: str,
    file_name: str,
    modality: str,
    desc: str,
    parsed_content: str,
    source_meta: dict | None = None,
) -> dict:
    """从「文件描述 + 内容解析」抽取记忆并归档到 memories（按角色隔离）。

    与 extract_and_archive 复用同一套 去重 → 向量化 → 摘要 → 归档 流程，
    仅抽取阶段换用文件专用 prompt。返回 {count, inserted_ids, summary}。
    """
    desc = (desc or "").strip()
    parsed_content = (parsed_content or "").strip()

    if not user_id or not (desc or parsed_content):
        return {"count": 0, "inserted_ids": [], "summary": ""}

    # 防幻觉：转录失败的占位文本（如 "[音频转录失败] xxx.mp3"）不能喂给 LLM，
    # 否则 LLM 会基于文件名编造"推测性内容"。当且仅当「desc 为空 + parsed_content
    # 是占位文本」时直接跳过；用户提供的 desc 仍可走抽取（描述本身也是有效信号）。
    if not desc and _is_placeholder_parsed_content(parsed_content):
        logger.info(
            "extract_from_file skip (transcribe failed placeholder, no desc)",
            extra=merge_extra(
                stage="extract_from_file",
                event="placeholder_skip",
                user_id=user_id,
                role_id=role_id,
                file_name=file_name,
                modality=modality,
                placeholder_preview=parsed_content[:80],
            ),
        )
        return {"count": 0, "inserted_ids": [], "summary": ""}

    settings = get_settings().memory
    t0 = time.perf_counter()

    # 1) 抽取
    items = await _llm_extract_file(file_name, modality, desc, parsed_content)
    if not items:
        logger.info(
            "extract_from_file skip (no items)",
            extra=merge_extra(
                stage="extract_from_file",
                event="empty",
                user_id=user_id,
                role_id=role_id,
                file_name=file_name,
                modality=modality,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
            ),
        )
        return {"count": 0, "inserted_ids": [], "summary": ""}

    # 2) 去重
    from embedding.bge_m3 import embed_texts

    facts = [it["fact"] for it in items]
    vectors = await asyncio.to_thread(embed_texts, facts)
    deduped_items: list[dict] = []
    skipped = 0
    for i, it in enumerate(items):
        v = vectors[i] if i < len(vectors) else []
        decision = await _dedup_one(user_id, it["fact"], v, settings.dedup_threshold, role_id)
        if decision.get("duplicate"):
            skipped += 1
            continue
        new_it = dict(it)
        new_it["fact"] = decision.get("merged") or it["fact"]
        rel = decision.get("relation") or it.get("relation") or ""
        if rel:
            new_it["relation"] = rel
        deduped_items.append(new_it)

    if not deduped_items:
        logger.info(
            "extract_from_file all duplicated",
            extra=merge_extra(
                stage="extract_from_file",
                event="all_dup",
                user_id=user_id,
                role_id=role_id,
                extracted=len(items),
                skipped=skipped,
            ),
        )
        return {"count": 0, "inserted_ids": [], "summary": ""}

    # 3) 向量化 + weaviate（EchoMemory）
    base_meta = {"source": "file_ingest", **(source_meta or {})}
    v_ids = await _vectorize_and_store(
        user_id, [it["fact"] for it in deduped_items], base_meta, role_id
    )

    # 4) 摘要 + 5) MySQL 归档
    summary = await _summarize(user_id, [it["fact"] for it in deduped_items])
    inserted = await _archive(user_id, deduped_items, v_ids, summary, role_id)
    logger.info(
        "extract_from_file done",
        extra=merge_extra(
            stage="extract_from_file",
            event="end",
            user_id=user_id,
            role_id=role_id,
            file_name=file_name,
            modality=modality,
            extracted=len(items),
            kept=len(deduped_items),
            skipped=skipped,
            inserted=len(inserted),
            summary_len=len(summary),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        ),
    )
    return {"count": len(inserted), "inserted_ids": inserted, "summary": summary}