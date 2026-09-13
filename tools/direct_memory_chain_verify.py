"""直接验证记忆链路3个改动 + 端到端链路:
1. ReAct=8 配置生效
2. L1+L2 parent_id 优先级合并
3. memory_relations 向量相似度因果源

跳过 UI 压测(LLM 速率限制导致 1000 轮不可行), 直接通过 echo-ai 服务接口验证。

输出:
- verification_results.json: 每项验证的结果
- 与 stress_logs.jsonl 合并用于生成报告
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("verify")

ECHO_CORE_BASE = "http://localhost:8080"
ECHO_AI_BASE = "http://localhost:8000"


@dataclass
class VerificationResult:
    name: str
    ok: bool
    detail: str = ""


# ---------- 1. ReAct=8 ----------

async def verify_react_max_iter() -> VerificationResult:
    from config.config import get_settings
    s = get_settings()
    actual = s.memory.react_max_iter
    return VerificationResult(
        name="ReAct max_iter 配置",
        ok=(actual == 8),
        detail=f"react_max_iter={actual} (期望 8)",
    )


# ---------- 2. L1+L2 parent_id 优先级 ----------

async def verify_l1_l2_parent_id() -> VerificationResult:
    """端到端: 注入 L2 + L1 children, 通过 /chat context 事件观察 l1_count 是否含父+子。"""
    # 注册测试用户
    user_id_resp = await _register_login(f"vfy_p_{uuid.uuid4().hex[:6]}")
    user_id = user_id_resp["user_id"]
    session_id = user_id_resp["session_id"]

    # 注入 L2 + 3 L1 children (parent_id = L2)
    from database import execute, fetch_one
    base = f"vfy_{uuid.uuid4().hex[:6]}"
    await execute(
        """INSERT INTO memories (user_id, role_id, level, content, summary, emotion_tag, emotion_intensity, parent_id)
        VALUES (%s, %s, 'L2', %s, %s, 'neutral', 0.0, NULL)""",
        (user_id, "default", f"{base}_L2_text", f"{base}_L2_summary"),
    )
    l2_row = await fetch_one("SELECT LAST_INSERT_ID() AS id")
    l2_id = int(l2_row["id"])
    l1_ids = []
    for i in range(3):
        await execute(
            """INSERT INTO memories (user_id, role_id, level, content, summary, emotion_tag, emotion_intensity, parent_id)
            VALUES (%s, %s, 'L1', %s, %s, 'neutral', 0.0, %s)""",
            (user_id, "default", f"{base}_L1_child_{i}", f"{base}_L1_child_{i}_summary", l2_id),
        )
        row = await fetch_one("SELECT LAST_INSERT_ID() AS id")
        l1_ids.append(int(row["id"]))

    # 直接调用 load_l1_summaries 验证
    from memory.retriever import load_l1_summaries
    out = await load_l1_summaries(str(user_id), role_id="default", limit=8)
    out_ids = {item["id"] for item in out}

    l2_in = l2_id in out_ids
    l1_in = any(cid in out_ids for cid in l1_ids)
    return VerificationResult(
        name="L1+L2 parent_id 优先级合并",
        ok=(l2_in and l1_in),
        detail=f"L2_id={l2_id} in_window={l2_in}; L1 children {l1_ids} in_window={l1_in}; total_count={len(out)}",
    )


# ---------- 3. memory_relations 向量相似度 ----------

async def verify_vector_cause() -> VerificationResult:
    """验证 _find_similar_cause 选中语义最相关的历史 fact。"""
    from database import execute, fetch_one
    from memory.extractor import _find_similar_cause
    from embedding.bge_m3 import embed_texts
    from vector.memory_store import get_memory_vector_store

    user_id = f"vfy_v_{uuid.uuid4().hex[:6]}"
    base = f"vc_{uuid.uuid4().hex[:6]}"

    facts = [
        f"{base}_A:用户在系统学习 Python 编程语言",
        f"{base}_B:用户计划明年去日本旅游",
        f"{base}_C:用户在练习钢琴,目标是能弹奏完整曲目",
    ]
    vecs = embed_texts(facts)
    inserted = []
    for fact in facts:
        await execute(
            """INSERT INTO memories (user_id, role_id, level, content, summary, emotion_tag, emotion_intensity, vector_id, embedding_dim)
            VALUES (%s, %s, 'L1', %s, %s, 'neutral', 0.0, %s, 1024)""",
            (user_id, "default", fact, fact, f"vfy-{uuid.uuid4().hex}"),
        )
        row = await fetch_one("SELECT LAST_INSERT_ID() AS id")
        inserted.append(int(row["id"]))

    vs = await asyncio.to_thread(get_memory_vector_store)
    await asyncio.to_thread(
        vs.add_texts,
        [f"mem-{uuid.uuid4().hex}" for _ in facts],
        facts,
        [{"userId": user_id, "roleId": "default", "memoryId": mid, "level": "L1"} for mid in inserted],
        vecs,
    )

    target_fact = f"{base}_target:用户说他在深入学习 Python,准备做开源项目"
    await execute(
        """INSERT INTO memories (user_id, role_id, level, content, summary, emotion_tag, emotion_intensity, vector_id, embedding_dim)
        VALUES (%s, %s, 'L1', %s, %s, 'neutral', 0.0, %s, 1024)""",
        (user_id, "default", target_fact, target_fact, f"vfy-{uuid.uuid4().hex}"),
    )
    row = await fetch_one("SELECT LAST_INSERT_ID() AS id")
    target_id = int(row["id"])

    target_vec = embed_texts([target_fact])[0]
    await asyncio.to_thread(
        vs.add_texts,
        [f"mem-{uuid.uuid4().hex}"],
        [target_fact],
        [{"userId": user_id, "roleId": "default", "memoryId": target_id, "level": "L1"}],
        [target_vec],
    )

    chosen = await _find_similar_cause(
        user_id, "default", target_fact, "L1", target_id, relation="extend",
    )
    correct = (chosen == inserted[0])
    return VerificationResult(
        name="memory_relations 向量相似度因果源",
        ok=correct,
        detail=f"chosen_id={chosen} expected={inserted[0]} (Python 学习主题); all_candidates={inserted}",
    )


# ---------- 4. 端到端 /chat 链路 ----------

async def _register_login(idx: int) -> dict:
    uname = f"vfy_{idx}_{uuid.uuid4().hex[:8]}"
    nick = f"验证{idx}"
    async with httpx.AsyncClient(base_url=ECHO_CORE_BASE, timeout=15.0) as c:
        r = await c.post("/api/auth/register",
                         json={"username": uname, "password": "test123", "nickname": nick})
        if r.json().get("code") not in (0, 200):
            raise RuntimeError(f"register failed: {r.text[:200]}")
        user_id = int(r.json()["data"]["id"])
        r = await c.post("/api/auth/login",
                         json={"username": uname, "password": "test123"})
        if r.json().get("code") not in (0, 200):
            raise RuntimeError(f"login failed: {r.text[:200]}")
        return {
            "user_id": user_id,
            "session_id": r.json()["data"]["sessionId"],
        }


async def verify_e2e_chat() -> VerificationResult:
    """端到端: 通过 /chat 触发 L0 抽取, 然后 DB 查询是否生成 L0 记忆。"""
    info = await _register_login(0)
    user_id = info["user_id"]
    session_id = info["session_id"]
    msg = f"我是 vfy_e2e_{uuid.uuid4().hex[:4]}, 是一名数据工程师, 喜欢用 Python 做后端开发"

    async with httpx.AsyncClient(base_url=ECHO_AI_BASE, timeout=60.0) as c:
        r = await c.post("/chat", json={
            "userId": str(user_id), "sessionId": session_id,
            "message": msg, "stream": False,
        })
        if r.status_code != 200:
            return VerificationResult(
                name="端到端 /chat 链路",
                ok=False,
                detail=f"chat returned {r.status_code}: {r.text[:200]}",
            )
        data = r.json()

    # 等记忆抽取完成(后台 fire-and-forget)
    await asyncio.sleep(15)

    from database import fetch_all
    mem_rows = await fetch_all(
        "SELECT level, content FROM memories WHERE user_id=%s ORDER BY id DESC LIMIT 5",
        (str(user_id),),
    )
    levels = [r["level"] for r in mem_rows]
    has_l0 = "L0" in levels
    reply = data.get("reply") or ""
    reply_nonempty = bool(reply.strip())

    return VerificationResult(
        name="端到端 /chat 链路 (L0 抽取)",
        ok=(has_l0 and reply_nonempty),
        detail=f"reply_len={len(reply)} reply_nonempty={reply_nonempty}; memories_generated={len(mem_rows)} levels={levels}; has_L0={has_l0}",
    )


# ---------- 5. 召回验证 ----------

async def verify_recall_quality() -> VerificationResult:
    """召回验证: 预注入 L0/L1, 发起查询, 检查 LLM 是否引用注入的记忆。"""
    info = await _register_login(0)
    user_id = info["user_id"]
    session_id = info["session_id"]

    from database import execute
    base = f"recall_{uuid.uuid4().hex[:6]}"

    l0_text = f"{base}_L0:用户是一名软件工程师"
    l1_text = f"{base}_L1:用户今天早上去了星巴克喝拿铁"
    await execute(
        """INSERT INTO memories (user_id, role_id, level, content, summary, emotion_tag, emotion_intensity)
        VALUES (%s, 'default', 'L0', %s, %s, 'neutral', 0.0)""",
        (str(user_id), l0_text, l0_text),
    )
    await execute(
        """INSERT INTO memories (user_id, role_id, level, content, summary, emotion_tag, emotion_intensity)
        VALUES (%s, 'default', 'L1', %s, %s, 'neutral', 0.0)""",
        (str(user_id), l1_text, l1_text),
    )

    issues: list[str] = []
    detail_parts: list[str] = []

    async def ask(question: str) -> tuple[str, list]:
        async with httpx.AsyncClient(base_url=ECHO_AI_BASE, timeout=60.0) as c:
            r = await c.post("/chat", json={
                "userId": str(user_id), "sessionId": session_id,
                "message": question, "stream": False,
            })
        if r.status_code != 200:
            return "", []
        d = r.json()
        return d.get("reply", ""), d.get("events", [])

    # Q1: L0 召回 — 职业
    reply1, events1 = await ask("我之前告诉过你我的职业是什么吗?")
    has_recall1 = any(e.get("type") == "memory_recall" for e in events1)
    hits1 = next((e for e in events1 if e.get("type") == "memory_recall"), {}).get("hits", [])
    q1_ok = ("软件工程师" in reply1 or "工程师" in reply1)
    detail_parts.append(
        f"Q1[职业]: reply_len={len(reply1)} mentions_l0={q1_ok} recall_event={has_recall1} hits={len(hits1)}"
    )
    if not q1_ok:
        issues.append("Q1[职业]: LLM 回复未引用注入的 L0 '软件工程师'")

    # Q2: L1 召回 — 事件
    reply2, events2 = await ask("我最近去过咖啡馆吗?")
    has_recall2 = any(e.get("type") == "memory_recall" for e in events2)
    hits2 = next((e for e in events2 if e.get("type") == "memory_recall"), {}).get("hits", [])
    q2_ok = ("星巴克" in reply2 or "拿铁" in reply2 or "咖啡" in reply2)
    detail_parts.append(
        f"Q2[咖啡馆]: reply_len={len(reply2)} mentions_l1={q2_ok} recall_event={has_recall2} hits={len(hits2)}"
    )
    if not q2_ok:
        issues.append("Q2[咖啡馆]: LLM 回复未引用注入的 L1 '星巴克/拿铁'")

    # Q3: cross-topic — 不应误命中
    reply3, _ = await ask("我之前养过什么宠物?")
    detail_parts.append(f"Q3[宠物]: reply_len={len(reply3)}")

    return VerificationResult(
        name="召回验证 (L0 职业 + L1 事件)",
        ok=not issues,
        detail=" | ".join(detail_parts) + (f" | ISSUES: {issues}" if issues else ""),
    )


# ---------- 主流程 ----------

async def main() -> None:
    logger.info("=" * 60)
    logger.info("直接验证记忆链路3个改动 + 端到端链路")
    logger.info("=" * 60)

    results = []

    logger.info("[1/4] ReAct=8 配置...")
    r = await verify_react_max_iter()
    results.append(r)
    logger.info("  %s — %s", "PASS" if r.ok else "FAIL", r.detail)

    logger.info("[2/4] L1+L2 parent_id 优先级合并...")
    r = await verify_l1_l2_parent_id()
    results.append(r)
    logger.info("  %s — %s", "PASS" if r.ok else "FAIL", r.detail)

    logger.info("[3/4] memory_relations 向量相似度...")
    r = await verify_vector_cause()
    results.append(r)
    logger.info("  %s — %s", "PASS" if r.ok else "FAIL", r.detail)

    logger.info("[4/5] 端到端 /chat 链路...")
    r = await verify_e2e_chat()
    results.append(r)
    logger.info("  %s — %s", "PASS" if r.ok else "FAIL", r.detail)

    logger.info("[5/5] 召回验证...")
    r = await verify_recall_quality()
    results.append(r)
    logger.info("  %s — %s", "PASS" if r.ok else "FAIL", r.detail)

    # 写结果
    out = Path("E:/AIWorking/workspace/report/echo/verification_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved -> %s", out)

    passed = sum(1 for r in results if r.ok)
    logger.info("=" * 60)
    logger.info("直接验证: %d/%d 通过", passed, len(results))

    # 输出问题清单
    for r in results:
        if not r.ok:
            logger.warning("问题: %s — %s", r.name, r.detail)


if __name__ == "__main__":
    asyncio.run(main())