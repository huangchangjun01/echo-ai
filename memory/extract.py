from __future__ import annotations

import json
import logging
from typing import Any

from config.config import get_settings
from llm.inference import call_llm
from memory.archiver import archive_memory, archive_relation, archive_summary
from memory.retrieve import retrieve_l1_memories

logger = logging.getLogger(__name__)


def _llm_config(model_size: str = "large") -> tuple[str, str, str, int, float]:
    """获取 LLM 配置，返回 (base_url, model, api_key, max_tokens, temperature)。"""
    settings = get_settings().llm
    if model_size == "small":
        cfg = settings.small
    else:
        cfg = settings.large
    return cfg.base_url, cfg.model, cfg.api_key, cfg.max_tokens, cfg.temperature


# ── 原子事实抽取 ───────────────────────────────────────────────────

_EXTRACT_ATOMIC_FACTS_PROMPT = """你是一个信息抽取专家。从以下对话中抽取原子事实和因果关系。

每条事实必须包含：
- fact: 事实内容（简洁的一句话）
- type: "atomic"（原子事实）或 "causal"（因果关系）
- 如果是因果关系，还需提供 source（原因事实）和 target（结果事实）

以 JSON 数组格式输出，只输出 JSON，不要包含其他内容。

对话消息：
{messages}

输出格式示例：
[{{"fact": "用户喜欢喝咖啡", "type": "atomic"}}, {{"fact": "喝咖啡导致失眠", "type": "causal", "source": "用户喝咖啡", "target": "用户失眠"}}]"""


async def extract_atomic_facts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """使用 LLM 从对话 messages 中抽取原子事实和因果关系。"""
    base_url, model, api_key, max_tokens, temperature = _llm_config("large")

    messages_text = json.dumps(messages, ensure_ascii=False, indent=2)
    prompt = _EXTRACT_ATOMIC_FACTS_PROMPT.format(messages=messages_text)

    response = await call_llm(
        base_url=base_url,
        model=model,
        api_key=api_key,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.get("choices", [{}])[0].get("message", {}).get("content", "[]")
    try:
        facts = json.loads(content)
        if isinstance(facts, list):
            logger.info("extract_atomic_facts: extracted %d facts", len(facts))
            return facts
    except json.JSONDecodeError:
        logger.warning("extract_atomic_facts: failed to parse LLM output: %s", content[:200])

    return []


# ── 向量化 ─────────────────────────────────────────────────────────

async def compute_embedding(text: str, modality: str = "text") -> list[float]:
    """根据 modality 选择 embedding 模型计算向量。

    modality: text → BGE-M3, image → CLIP, audio → Whisper
    """
    if modality == "text":
        try:
            from embedding.bge_m3 import compute_query_embedding
            return compute_query_embedding(text)
        except Exception:
            logger.warning("BGE-M3 embedding failed, falling back to Chinese-CLIP")
            from embedding.models import compute_text_embedding
            return compute_text_embedding(text)
    elif modality == "image":
        from embedding.models import compute_image_embeddings
        vecs = compute_image_embeddings([text.encode("utf-8")])
        return vecs[0] if vecs else []
    elif modality == "audio":
        try:
            from embedding.whisper import extract_voiceprint
            return extract_voiceprint(text)
        except Exception:
            logger.warning("Whisper embedding failed, falling back to BGE-M3")
            from embedding.bge_m3 import compute_query_embedding
            return compute_query_embedding(text)
    else:
        from embedding.bge_m3 import compute_query_embedding
        return compute_query_embedding(text)


# ── 语义去重 ───────────────────────────────────────────────────────

_DEDUP_PROMPT = """你是一个语义去重专家。判断以下两个事实是否表达相同或高度相似的含义。

事实 A: {fact_a}
事实 B: {fact_b}

如果是重复/高度相似，回答 JSON: {{"is_duplicate": true, "reason": "简要说明"}}
如果不重复，回答 JSON: {{"is_duplicate": false, "reason": "简要说明"}}

只输出 JSON。"""


async def semantic_dedup(facts: list[dict[str, Any]], user_id: str) -> list[dict[str, Any]]:
    """对 facts 进行语义去重：向量化 + L1 检索 + LLM 判重。"""
    settings = get_settings().memory
    deduped: list[dict[str, Any]] = []

    for fact in facts:
        fact_text = fact.get("fact", "")
        if not fact_text:
            continue

        # 向量化当前 fact
        try:
            query_vector = await compute_embedding(fact_text, modality="text")
        except Exception as e:
            logger.warning("semantic_dedup: embedding failed for fact=%s: %s", fact_text[:50], e)
            deduped.append(fact)
            continue

        # 检索 L1 相似记忆
        try:
            similar = await retrieve_l1_memories(user_id, fact_text, k=5)
        except Exception as e:
            logger.warning("semantic_dedup: retrieve failed for fact=%s: %s", fact_text[:50], e)
            deduped.append(fact)
            continue

        is_duplicate = False
        for sim in similar:
            similarity = sim.get("metadata", {}).get("similarity", 0)
            if similarity >= settings.dedup_threshold:
                # 调用 LLM 判重
                base_url, model, api_key, max_tokens, temperature = _llm_config("small")
                prompt = _DEDUP_PROMPT.format(
                    fact_a=fact_text,
                    fact_b=sim.get("content", ""),
                )
                try:
                    response = await call_llm(
                        base_url=base_url,
                        model=model,
                        api_key=api_key,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=128,
                    )
                    content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                    result = json.loads(content)
                    if result.get("is_duplicate"):
                        logger.info("semantic_dedup: duplicate fact=%s", fact_text[:50])
                        is_duplicate = True
                        break
                except Exception as e:
                    logger.warning("semantic_dedup: LLM dedup check failed: %s", e)

        if not is_duplicate:
            deduped.append(fact)

    logger.info("semantic_dedup: %d → %d facts", len(facts), len(deduped))
    return deduped


# ── 关系识别 ───────────────────────────────────────────────────────

_RELATION_PROMPT = """你是一个知识关系识别专家。分析新事实与已有记忆之间的关系。

新事实: {new_fact}

已有记忆列表:
{existing_memories}

请识别新事实与每条已有记忆的关系类型，可选类型：
- "causes": 新事实导致了已有记忆中的描述
- "update": 新事实更新/修正了已有记忆
- "contradict": 新事实与已有记忆矛盾
- "extend": 新事实扩展/补充了已有记忆

以 JSON 数组格式输出，只输出 JSON：
[{{"type": "causes", "target_id": 1, "confidence": 0.85}}, ...]"""


async def identify_relations(
    new_fact: dict[str, Any],
    existing_memories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """使用 LLM 识别新记忆与已有记忆的关系。"""
    if not existing_memories:
        return []

    fact_text = new_fact.get("fact", "")
    if not fact_text:
        return []

    # 构建已有记忆的简要描述
    mem_desc = json.dumps(
        [{"id": m.get("id"), "content": m.get("content", "")[:200]} for m in existing_memories],
        ensure_ascii=False,
        indent=2,
    )

    base_url, model, api_key, max_tokens, temperature = _llm_config("large")
    prompt = _RELATION_PROMPT.format(new_fact=fact_text, existing_memories=mem_desc)

    try:
        response = await call_llm(
            base_url=base_url,
            model=model,
            api_key=api_key,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "[]")
        relations = json.loads(content)
        if isinstance(relations, list):
            logger.info("identify_relations: identified %d relations", len(relations))
            return relations
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("identify_relations: failed: %s", e)

    return []


# ── 情感标注 ───────────────────────────────────────────────────────

_EMOTION_PROMPT = """你是一个情感分析专家。对以下事实进行情感标注。

事实: {fact}

请输出 JSON，包含：
- emotion_tag: 情感标签（joy/sadness/anger/fear/surprise/disgust/neutral/anticipation/trust）
- intensity: 情感强度（0.0-1.0）

只输出 JSON，例如：{{"emotion_tag": "joy", "intensity": 0.8}}"""


async def tag_emotion(fact: dict[str, Any]) -> dict[str, Any]:
    """使用 LLM 对事实进行情感标注。"""
    fact_text = fact.get("fact", "")
    if not fact_text:
        fact["emotion_tag"] = "neutral"
        fact["intensity"] = 0.0
        return fact

    base_url, model, api_key, max_tokens, temperature = _llm_config("small")
    prompt = _EMOTION_PROMPT.format(fact=fact_text)

    try:
        response = await call_llm(
            base_url=base_url,
            model=model,
            api_key=api_key,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=64,
        )
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        emotion = json.loads(content)
        fact["emotion_tag"] = emotion.get("emotion_tag", "neutral")
        fact["intensity"] = emotion.get("intensity", 0.0)
        logger.info("tag_emotion: fact=%s tag=%s intensity=%.2f", fact_text[:50], fact["emotion_tag"], fact["intensity"])
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("tag_emotion: failed: %s", e)
        fact["emotion_tag"] = "neutral"
        fact["intensity"] = 0.0

    return fact


# ── 记忆合并 ───────────────────────────────────────────────────────

_MERGE_PROMPT = """你是一个知识合并专家。将以下两条相似记忆合并为一条简洁、完整的记忆。

记忆 A: {mem_a}
记忆 B: {mem_b}

请输出合并后的记忆文本，只输出文本内容，不要包含其他内容。"""


async def merge_similar_memories(user_id: str) -> None:
    """查询相似度 > merge_threshold 的记忆，使用 LLM 合并，删除旧记忆保留合并后的。"""
    settings = get_settings().memory

    from database.mysql import get_session
    from database.models import CoreMemory
    from sqlalchemy import select

    async with get_session() as session:
        result = await session.execute(
            select(CoreMemory)
            .where(CoreMemory.user_id == user_id)
            .order_by(CoreMemory.id.asc())
        )
        all_memories = result.scalars().all()

    if len(all_memories) < 2:
        return

    # 对每对记忆进行相似度检查
    merged_ids: set[int] = set()
    for i, mem_a in enumerate(all_memories):
        if mem_a.id in merged_ids:
            continue
        for mem_b in all_memories[i + 1 :]:
            if mem_b.id in merged_ids:
                continue
            # 使用向量相似度判断
            try:
                vec_a = await compute_embedding(mem_a.content, modality="text")
                vec_b = await compute_embedding(mem_b.content, modality="text")
                similarity = _cosine_similarity(vec_a, vec_b)
            except Exception:
                continue

            if similarity >= settings.merge_threshold:
                # LLM 合并
                base_url, model, api_key, max_tokens, temperature = _llm_config("large")
                prompt = _MERGE_PROMPT.format(mem_a=mem_a.content, mem_b=mem_b.content)
                try:
                    response = await call_llm(
                        base_url=base_url,
                        model=model,
                        api_key=api_key,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    merged_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if merged_content:
                        # 归档合并后的记忆
                        await archive_memory(
                            user_id=user_id,
                            content=merged_content,
                            memory_type=mem_a.memory_type,
                            importance=max(mem_a.importance, mem_b.importance),
                            emotion_tag=mem_a.emotion_tag,
                            emotion_intensity=max(mem_a.emotion_intensity, mem_b.emotion_intensity),
                            weaviate_uuid=None,
                            embedding_dim=mem_a.embedding_dim,
                            summary=None,
                        )
                        # 标记旧记忆为待删除
                        merged_ids.add(mem_a.id)
                        merged_ids.add(mem_b.id)
                        logger.info(
                            "merge_similar_memories: merged id=%s and id=%s similarity=%.3f",
                            mem_a.id, mem_b.id, similarity,
                        )
                except Exception as e:
                    logger.warning("merge_similar_memories: LLM merge failed: %s", e)

    # 删除旧记忆
    if merged_ids:
        from sqlalchemy import delete
        async with get_session() as session:
            await session.execute(
                delete(CoreMemory).where(CoreMemory.id.in_(list(merged_ids)))
            )
        logger.info("merge_similar_memories: deleted %d old memories for user=%s", len(merged_ids), user_id)


# ── 矛盾清理 ───────────────────────────────────────────────────────

_CONTRADICT_PROMPT = """你是一个矛盾检测专家。判断以下两条记忆是否存在事实或情感冲突。

记忆 A: {mem_a}
记忆 B: {mem_b}

如果存在矛盾，回答 JSON: {{"is_contradiction": true, "keep": "A", "reason": "简要说明"}}
其中 "keep" 指定应保留哪条（A 或 B）。
如果不存在矛盾，回答 JSON: {{"is_contradiction": false}}

只输出 JSON。"""


async def resolve_contradictions(user_id: str) -> None:
    """查询相似但情感/事实冲突的记忆，使用 LLM 判断矛盾并清理。"""
    settings = get_settings().memory

    from database.mysql import get_session
    from database.models import CoreMemory
    from sqlalchemy import select, delete

    async with get_session() as session:
        result = await session.execute(
            select(CoreMemory)
            .where(CoreMemory.user_id == user_id)
            .order_by(CoreMemory.id.asc())
        )
        all_memories = result.scalars().all()

    if len(all_memories) < 2:
        return

    to_delete: set[int] = set()
    for i, mem_a in enumerate(all_memories):
        if mem_a.id in to_delete:
            continue
        for mem_b in all_memories[i + 1 :]:
            if mem_b.id in to_delete:
                continue
            try:
                vec_a = await compute_embedding(mem_a.content, modality="text")
                vec_b = await compute_embedding(mem_b.content, modality="text")
                similarity = _cosine_similarity(vec_a, vec_b)
            except Exception:
                continue

            if similarity >= settings.contradiction_threshold:
                base_url, model, api_key, max_tokens, temperature = _llm_config("small")
                prompt = _CONTRADICT_PROMPT.format(mem_a=mem_a.content, mem_b=mem_b.content)
                try:
                    response = await call_llm(
                        base_url=base_url,
                        model=model,
                        api_key=api_key,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=128,
                    )
                    content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                    result = json.loads(content)
                    if result.get("is_contradiction"):
                        keep = result.get("keep", "A")
                        if keep == "A":
                            to_delete.add(mem_b.id)
                        else:
                            to_delete.add(mem_a.id)
                        logger.info(
                            "resolve_contradictions: contradiction between %s and %s, keeping %s",
                            mem_a.id, mem_b.id, keep,
                        )
                except Exception as e:
                    logger.warning("resolve_contradictions: LLM check failed: %s", e)

    if to_delete:
        async with get_session() as session:
            await session.execute(
                delete(CoreMemory).where(CoreMemory.id.in_(list(to_delete)))
            )
        logger.info("resolve_contradictions: deleted %d contradictory memories for user=%s", len(to_delete), user_id)


# ── 摘要生成 ───────────────────────────────────────────────────────

_SUMMARY_PROMPT = """你是一个记忆摘要专家。根据以下用户的 L0 核心记忆，生成一份简洁的记忆摘要。

核心记忆列表：
{memories}

摘要要求：
- 简洁清晰，不超过 {max_length} 字
- 涵盖关键信息、偏好、重要事件
- 以第三人称叙述

只输出摘要文本，不要包含其他内容。"""


async def generate_summary(user_id: str) -> str:
    """查询用户所有 L0 记忆，使用 LLM 生成记忆摘要并保存。"""
    from memory.archiver import get_l0_memories

    settings = get_settings().memory
    l0_memories = await get_l0_memories(user_id)

    if not l0_memories:
        logger.info("generate_summary: no L0 memories for user=%s", user_id)
        return ""

    memory_texts = "\n".join(
        f"- {m['content']}" for m in l0_memories
    )

    base_url, model, api_key, max_tokens, temperature = _llm_config("large")
    prompt = _SUMMARY_PROMPT.format(
        memories=memory_texts,
        max_length=settings.max_summary_length,
    )

    try:
        response = await call_llm(
            base_url=base_url,
            model=model,
            api_key=api_key,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        summary = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        summary = summary.strip()
    except Exception as e:
        logger.error("generate_summary: LLM call failed: %s", e)
        return ""

    if summary:
        memory_ids = ",".join(str(m["id"]) for m in l0_memories)
        await archive_summary(user_id, summary, memory_ids)
        logger.info("generate_summary: summary generated for user=%s len=%d", user_id, len(summary))

    return summary


# ── 辅助函数 ───────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两条向量的余弦相似度。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── 主入口 ─────────────────────────────────────────────────────────

async def extract_and_archive(
    user_id: str,
    messages: list[dict[str, Any]],
    background_tasks: Any = None,
) -> dict[str, Any]:
    """综合执行记忆抽取流水线：抽取 → 去重 → 向量化 → 归档 → 关系识别 → 情感标注 → 合并清理 → 摘要。

    返回 {"archived_count": ..., "relations_count": ..., "summary": "..."}
    """
    archived_count = 0
    relations_count = 0
    summary_text = ""

    # 1. 抽取原子事实
    facts = await extract_atomic_facts(messages)
    if not facts:
        logger.info("extract_and_archive: no facts extracted for user=%s", user_id)
        return {"archived_count": 0, "relations_count": 0, "summary": ""}

    # 2. 语义去重
    deduped_facts = await semantic_dedup(facts, user_id)

    # 3. 加载已有 L0 记忆（用于关系识别）
    from memory.archiver import get_l0_memories
    existing_memories = await get_l0_memories(user_id)

    # 4. 逐条处理：向量化 → 归档 → 关系识别 → 情感标注
    archived_ids: list[int] = []
    for fact in deduped_facts:
        fact_text = fact.get("fact", "")
        fact_type = fact.get("type", "atomic")
        if not fact_text:
            continue

        # 情感标注
        fact = await tag_emotion(fact)

        # 向量化
        try:
            embedding = await compute_embedding(fact_text, modality="text")
        except Exception as e:
            logger.warning("extract_and_archive: embedding failed for fact=%s: %s", fact_text[:50], e)
            embedding = []

        # 计算重要性：基于 fact_type 和情感强度
        importance = 0.5
        if fact_type == "causal":
            importance = 0.7
        importance = min(1.0, importance + fact.get("intensity", 0.0) * 0.3)

        # 归档到 MySQL
        memory_id = await archive_memory(
            user_id=user_id,
            content=fact_text,
            memory_type="text",
            importance=importance,
            emotion_tag=fact.get("emotion_tag"),
            emotion_intensity=fact.get("intensity", 0.0),
            weaviate_uuid=None,
            embedding_dim=len(embedding),
            summary=None,
        )
        archived_ids.append(memory_id)
        archived_count += 1

        # 关系识别
        relations = await identify_relations(fact, existing_memories[:10])
        for rel in relations:
            target_id = rel.get("target_id")
            if target_id and target_id in [m["id"] for m in existing_memories]:
                await archive_relation(
                    source_id=memory_id,
                    target_id=target_id,
                    relation_type=rel.get("type", "extend"),
                    confidence=rel.get("confidence", 0.5),
                )
                relations_count += 1

    # 5. 合并相似记忆（后台任务）
    if background_tasks is not None:
        background_tasks.add_task(merge_similar_memories, user_id)
        background_tasks.add_task(resolve_contradictions, user_id)
    else:
        try:
            await merge_similar_memories(user_id)
        except Exception as e:
            logger.warning("extract_and_archive: merge_similar_memories failed: %s", e)
        try:
            await resolve_contradictions(user_id)
        except Exception as e:
            logger.warning("extract_and_archive: resolve_contradictions failed: %s", e)

    # 6. 生成摘要
    try:
        summary_text = await generate_summary(user_id)
    except Exception as e:
        logger.warning("extract_and_archive: generate_summary failed: %s", e)

    logger.info(
        "extract_and_archive: done user=%s archived=%d relations=%d",
        user_id, archived_count, relations_count,
    )
    return {
        "archived_count": archived_count,
        "relations_count": relations_count,
        "summary": summary_text,
    }