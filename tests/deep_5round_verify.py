"""5 轮深度对话验证测试。

设计 (单用户单线程, 避免 LLM 速率干扰):
  R1: 多条 L0 偏好陈述(身份 + 宠物 + 公司)
  R2: L1 近期事件叙述(技术细节)
  R3: 另一条相关 L1 (可与 R2 触发因果关系)
  R4: 矛盾更新 (反转 R1 的 L0 偏好)
  R5: 主动召回查询 (验证 LLM 能引用前面的记忆)

每轮:
  1. 调 /chat (stream=False) 等完整响应
  2. 等待 30s 让 extract_and_archive 后台完成
  3. 立即查询 DB 看是否生成新记忆/关系

测试结束后:
  - 汇总 5 轮的抽取数、关系数
  - 评估每轮的抽取质量(关键词命中)
  - 评估召回质量(R5 的回复是否引用 R1/R2/R3 内容)
  - 输出 Markdown 报告

用法: python -m tests.deep_5round_verify
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("deep5")

ECHO_CORE_BASE = "http://localhost:8080"
ECHO_AI_BASE = "http://localhost:8000"


@dataclass
class RoundSpec:
    name: str
    user_message: str
    # 期望抽取关键词 (用于评估 L0/L1 是否覆盖)
    expect_keywords: list[str]
    # 期望触发的 relation (causes / update / extend / contradict / None)
    expect_relation: str | None = None


@dataclass
class RoundResult:
    round_id: int
    name: str
    user_message: str
    reply: str
    reply_keywords_found: list[str] = field(default_factory=list)
    new_memories: list[dict] = field(default_factory=list)
    new_relations: list[dict] = field(default_factory=list)
    extract_log_status: str = ""
    extraction_ok: bool = False
    issues: list[str] = field(default_factory=list)


ROUND_SPECS: list[RoundSpec] = [
    RoundSpec(
        name="R1: 多条 L0 偏好陈述",
        user_message=(
            "我想跟你分享一下我自己。"
            "我是一名 Python 后端工程师, 工作 5 年了, 一直在做 AI 相关的产品。"
            "我在一家叫 Echo 的公司上班, 团队大概 30 个人。"
            "我养了一只小狗, 叫小黑, 已经 3 岁了。"
        ),
        expect_keywords=["Python", "后端工程师", "AI", "Echo", "小狗", "小黑"],
    ),
    RoundSpec(
        name="R2: L1 近期事件(技术细节)",
        user_message=(
            "今天上午我在公司部署了一个新的 AI 服务, "
            "用的是 FastAPI 框架 + BGE-M3 做 embedding, "
            "上线后响应延迟从 800ms 降到了 200ms。"
        ),
        expect_keywords=["FastAPI", "BGE-M3", "embedding", "200ms", "延迟"],
    ),
    RoundSpec(
        name="R3: 相关 L1(可能触发 causes/extend 关系)",
        user_message=(
            "我最近开始学 Rust, 因为想把那个 AI 服务的核心调度模块重写, "
            "提升一下并发性能, Python 的 GIL 一直让我头疼。"
        ),
        expect_keywords=["Rust", "AI 服务", "GIL", "并发", "重写"],
        expect_relation="causes",
    ),
    RoundSpec(
        name="R4: 矛盾更新(反转 R1 的 L0 偏好)",
        user_message=(
            "其实我最近想跟你说, 我有点不太喜欢 Python 了, "
            "感觉 Rust 写后端更优雅, 类型系统也更强。"
            "我准备转 Rust 了。"
        ),
        expect_keywords=["不喜欢 Python", "Rust", "转 Rust"],
        expect_relation="contradict",
    ),
    RoundSpec(
        name="R5: 主动召回查询",
        user_message=(
            "你还记得我之前跟你聊过我的工作吗? "
            "我养了什么宠物? 我最近在学什么新语言?"
        ),
        expect_keywords=["Python", "后端", "Echo", "小狗", "小黑", "Rust"],
    ),
]


async def register_login(idx: int) -> dict:
    """注册 + 登录,返回 user_id / session_id"""
    uname = f"deep5_{idx}_{uuid.uuid4().hex[:8]}"
    nick = f"深度测试{idx}"
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


async def query_db(user_id: str, round_id: int) -> tuple[list, list, str]:
    """查询本轮新增的 memories / relations / extract log status"""
    from database import fetch_all, fetch_one

    # 用 created_at > 测试启动时间作为本轮新增的判定
    cutoff_ts = (datetime.now().timestamp() - 600)  # 最近 10 分钟

    mem_rows = await fetch_all(
        """SELECT id, level, content, parent_id, created_at
        FROM memories WHERE user_id=%s AND created_at > FROM_UNIXTIME(%s)
        ORDER BY id ASC""",
        (user_id, cutoff_ts),
    )
    mem_list = [{
        "id": r["id"], "level": r["level"],
        "content": (r.get("content") or "")[:120],
        "parent_id": r.get("parent_id"),
    } for r in mem_rows]

    # relations
    rel_rows = await fetch_all(
        """SELECT r.id, r.source_id, r.target_id, r.relation,
               ms.content AS source_content, mt.content AS target_content
        FROM memory_relations r
        LEFT JOIN memories ms ON ms.id = r.source_id
        LEFT JOIN memories mt ON mt.id = r.target_id
        WHERE r.user_id=%s AND r.created_at > FROM_UNIXTIME(%s)
        ORDER BY r.id ASC""",
        (user_id, cutoff_ts),
    )
    rel_list = [{
        "id": r["id"], "source_id": r["source_id"], "target_id": r["target_id"],
        "relation": r["relation"],
        "source_content": (r.get("source_content") or "")[:80],
        "target_content": (r.get("target_content") or "")[:80],
    } for r in rel_rows]

    # extract log status (本用户最近一条)
    log_row = await fetch_one(
        """SELECT status, error FROM memory_extract_logs
        WHERE user_id=%s ORDER BY id DESC LIMIT 1""",
        (user_id,),
    )
    log_status = ""
    if log_row:
        log_status = log_row["status"]
        if log_row.get("error"):
            log_status += f" (err: {log_row['error'][:60]})"
    return mem_list, rel_list, log_status


async def wait_for_extract_complete(user_id: str, timeout: int = 120) -> str:
    """等待 memory_extract_logs 最后一条 status 变为 success/failed"""
    from database import fetch_one

    t0 = time.perf_counter()
    last_status = ""
    while time.perf_counter() - t0 < timeout:
        row = await fetch_one(
            """SELECT status FROM memory_extract_logs
            WHERE user_id=%s ORDER BY id DESC LIMIT 1""",
            (user_id,),
        )
        if row:
            status = row["status"]
            if status in ("success", "failed"):
                return status
            last_status = status
        await asyncio.sleep(2)
    return last_status or "timeout"


def evaluate_round(spec: RoundSpec, reply: str, mem_list: list[dict],
                   rel_list: list[dict], extract_status: str) -> RoundResult:
    """评估一轮"""
    issues: list[str] = []

    # 1. reply 关键词命中
    found = [kw for kw in spec.expect_keywords if kw in reply]
    # R5 是召回查询,回复里没关键词也算正常(LLM 可能用同义词)
    if spec.name.startswith("R5"):
        reply_kw_ok = len(found) >= 2  # 召回至少命中 2 个关键词
    else:
        # R1-R4 是输入陈述,LLM 回复不一定回显关键词
        # 这里只统计,不算 fail
        reply_kw_ok = True

    # 2. 抽取状态
    extraction_ok = (extract_status == "success")
    if extract_status != "success":
        if extract_status == "pending":
            issues.append(f"抽取 pending 超时(LLM 慢), extract_status={extract_status}")
        elif extract_status == "failed":
            issues.append(f"抽取失败, extract_status={extract_status}")
        elif extract_status == "timeout":
            issues.append("等抽取状态超时(120s), 可能还在跑")

    # 3. 记忆生成检查
    if not mem_list:
        issues.append(f"本轮未生成任何记忆")
    else:
        levels = [m["level"] for m in mem_list]
        # R1 应有 L0
        if spec.name.startswith("R1"):
            if "L0" not in levels:
                issues.append(f"R1 应有 L0, 实际 levels={levels}")
        # R2/R3 应有 L1
        if spec.name.startswith("R2") or spec.name.startswith("R3"):
            if "L1" not in levels:
                issues.append(f"{spec.name} 应有 L1, 实际 levels={levels}")
        # R4 应有 L0 (反转陈述)
        if spec.name.startswith("R4"):
            # 应有 contradict 关系
            contradict_rels = [r for r in rel_list if r["relation"] == "contradict"]
            if not contradict_rels:
                issues.append(f"R4 应触发 contradict 关系, 实际关系={[r['relation'] for r in rel_list]}")

    # 4. 关系检查
    if spec.expect_relation:
        rels_of_type = [r for r in rel_list if r["relation"] == spec.expect_relation]
        if not rels_of_type:
            issues.append(
                f"期望关系类型 {spec.expect_relation}, 实际类型 {[r['relation'] for r in rel_list]}"
            )

    # 5. R5 召回评估: LLM 应引用前面的事实
    if spec.name.startswith("R5"):
        # 至少 2 个关键词被 LLM 回复引用
        if len(found) < 2:
            issues.append(
                f"R5 召回关键词命中不足: 期望≥2, 实际 {len(found)} 个 ({found})"
            )

    return RoundResult(
        round_id=0,
        name=spec.name,
        user_message=spec.user_message,
        reply=reply[:600],
        reply_keywords_found=found,
        new_memories=mem_list,
        new_relations=rel_list,
        extract_log_status=extract_status,
        extraction_ok=extraction_ok,
        issues=issues,
    )


async def run_round(round_id: int, spec: RoundSpec, user_id: str,
                    session_id: str) -> RoundResult:
    logger.info("=" * 60)
    logger.info("[R%d] %s", round_id, spec.name)
    logger.info("  user_message: %s...", spec.user_message[:100])

    # 1) 调 /chat
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(base_url=ECHO_AI_BASE, timeout=120.0) as c:
            r = await c.post("/chat", json={
                "userId": str(user_id),
                "sessionId": session_id,
                "message": spec.user_message,
                "stream": False,
            })
            if r.status_code != 200:
                logger.error("  chat failed: %s", r.text[:200])
                return RoundResult(
                    round_id=round_id, name=spec.name,
                    user_message=spec.user_message, reply="",
                    extract_log_status="chat_failed",
                    extraction_ok=False,
                    issues=[f"chat returned {r.status_code}"],
                )
            data = r.json()
            reply = data.get("reply") or ""
    except Exception as e:
        logger.error("  chat exception: %s", e)
        return RoundResult(
            round_id=round_id, name=spec.name,
            user_message=spec.user_message, reply="",
            extract_log_status="exception",
            extraction_ok=False,
            issues=[f"chat exception: {str(e)[:100]}"],
        )
    chat_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("  chat reply: %s (latency=%dms)", reply[:60].replace("\n", " "), chat_ms)

    # 2) 等抽取完成 (后台 fire-and-forget)
    extract_status = await wait_for_extract_complete(str(user_id), timeout=120)
    logger.info("  extract status: %s", extract_status)

    # 3) 立即查 DB
    mem_list, rel_list, log_status = await query_db(str(user_id), round_id)

    # 4) 评估
    result = evaluate_round(spec, reply, mem_list, rel_list, extract_status)
    result.round_id = round_id
    if result.issues:
        logger.warning("  ISSUES: %s", result.issues)
    else:
        logger.info("  ✓ OK")
    return result


def render_report(results: list[RoundResult]) -> str:
    lines: list[str] = []
    lines.append("# Echo 记忆链路 5 轮深度验证报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 测试设计")
    lines.append("- 单用户单线程,避免 LLM 速率干扰")
    lines.append("- 5 轮对话: 1 偏好陈述 / 1 事件叙述 / 1 相关事件 / 1 矛盾更新 / 1 召回查询")
    lines.append("- 每轮调 `/chat`,等回复;再等 `extract_and_archive` 后台完成 (最长 120s);查 DB 验证")
    lines.append("")

    # 总览
    total_issues = sum(len(r.issues) for r in results)
    rounds_ok = sum(1 for r in results if not r.issues)
    lines.append("## 一、总览")
    lines.append(f"- 总轮次: **{len(results)}**")
    lines.append(f"- 通过轮次 (无问题): **{rounds_ok}/{len(results)}**")
    lines.append(f"- 累计问题: **{total_issues}**")
    lines.append("")
    lines.append(f"{( '✅ 全部轮次通过' if rounds_ok == len(results) else f'❌ {len(results) - rounds_ok} 轮存在问题' )}")
    lines.append("")

    # 逐轮详情
    lines.append("## 二、逐轮详情")
    lines.append("")
    for r in results:
        lines.append(f"### {r.name}")
        lines.append("")
        lines.append(f"**用户消息**: {r.user_message[:200]}...")
        lines.append("")
        lines.append(f"**LLM 回复**: {r.reply[:200]}...")
        lines.append("")
        lines.append(f"**回复关键词命中**: {r.reply_keywords_found}")
        lines.append("")
        lines.append(f"**抽取状态**: `{r.extract_log_status}` (ok={r.extraction_ok})")
        lines.append("")
        lines.append(f"**本轮新增记忆**: {len(r.new_memories)} 条")
        if r.new_memories:
            for m in r.new_memories:
                lines.append(f"  - id={m['id']} level={m['level']} parent={m['parent_id']} content=`{m['content']}`")
        else:
            lines.append("  - (无)")
        lines.append("")
        lines.append(f"**本轮新增关系**: {len(r.new_relations)} 条")
        if r.new_relations:
            for rel in r.new_relations:
                lines.append(
                    f"  - id={rel['id']} {rel['relation']}: "
                    f"src({rel['source_content'][:40]}) → tgt({rel['target_content'][:40]})"
                )
        else:
            lines.append("  - (无)")
        lines.append("")
        if r.issues:
            lines.append("**问题**:")
            for issue in r.issues:
                lines.append(f"  - ❌ {issue}")
        else:
            lines.append("**问题**: (无)")
        lines.append("")
        lines.append("---")
        lines.append("")

    # 问题汇总
    all_issues: list[tuple[str, str]] = []
    for r in results:
        for issue in r.issues:
            all_issues.append((r.name, issue))
    if all_issues:
        lines.append("## 三、问题汇总")
        lines.append("")
        for name, issue in all_issues:
            lines.append(f"- **{name}**: {issue}")
        lines.append("")

    # 结论
    lines.append("## 四、结论")
    if rounds_ok == len(results):
        lines.append("✅ **抽取、召回、因果链均符合预期**。3 个改动(ReAct=8, L1/L2 parent_id, memory_relations 向量相似度)在生产路径中验证生效。")
    else:
        lines.append(f"⚠️ **{len(results) - rounds_ok} 轮存在问题**,需进一步修复:")
        # 关键观察
        lines.append("")
        lines.append("### 关键观察")
        # 抽取失败规律
        ext_failures = [r for r in results if r.extract_log_status != "success"]
        if ext_failures:
            lines.append(f"- **抽取稳定性**: {len(ext_failures)}/{len(results)} 轮抽取未在 120s 内变为 success")
        # 关系建立规律
        no_relations = [r for r in results if r.name.startswith(("R3", "R4")) and not r.new_relations]
        if no_relations:
            lines.append(f"- **因果链建立**: R3/R4 应触发关系但实际未生成")
        # 召回效果
        r5 = next((r for r in results if r.name.startswith("R5")), None)
        if r5:
            lines.append(f"- **召回质量**: R5 关键词命中 {len(r5.reply_keywords_found)} 个")

    return "\n".join(lines)


async def main() -> None:
    logger.info("=" * 60)
    logger.info("5 轮深度对话验证测试 (单用户单线程)")
    logger.info("=" * 60)

    # 1) 注册测试用户
    auth = await register_login(0)
    user_id = auth["user_id"]
    session_id = auth["session_id"]
    logger.info("test user_id=%s session_id=%s", user_id, session_id[:12])

    # 2) 逐轮执行
    results: list[RoundResult] = []
    for i, spec in enumerate(ROUND_SPECS, start=1):
        result = await run_round(i, spec, user_id, session_id)
        results.append(result)

    # 3) 渲染 + 保存报告
    md = render_report(results)
    out_dir = Path("E:/AIWorking/workspace/report/echo")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "deep_5round_report.md"
    md_path.write_text(md, encoding="utf-8")
    logger.info("MD report -> %s", md_path)

    # 4) JSON 明细
    json_path = out_dir / "deep_5round_results.json"
    json_path.write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("JSON results -> %s", json_path)

    # 5) 总结
    total_issues = sum(len(r.issues) for r in results)
    logger.info("=" * 60)
    logger.info("总结: %d/%d 通过,累计 %d 个问题",
                sum(1 for r in results if not r.issues),
                len(results), total_issues)


if __name__ == "__main__":
    asyncio.run(main())