"""验证报告生成器:基于压测日志和数据库状态,评估记忆链路质量。

验证维度:
1. 链路正确性:每轮是否收到 assistant 回复
2. 召回及时性:响应延迟分位数
3. 抽取合理性:对每类对话,检查 DB 是否正确生成对应层级记忆
4. 关系识别:contradict 类是否触发矛盾边,extend/update 是否被识别
5. 异常轮次清单: 空回复/超时/错误的分布

输入:
  - stress_logs.jsonl (压测明细)
  - stress_metrics.json (压测聚合)
  - MySQL (memories, memory_relations, memory_extract_logs)

输出:
  - Markdown 报告
  - PDF (转给 echo 复盘)
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")


@dataclass
class ValidationResult:
    section: str
    metric: str
    value: Any
    status: str  # ok | warn | fail
    note: str = ""


def load_logs(log_path: str) -> list[dict]:
    logs = []
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            logs.append(json.loads(line))
    return logs


async def db_stats() -> dict[str, Any]:
    from database import fetch_all

    out: dict[str, Any] = {}

    rows = await fetch_all(
        "SELECT user_id, level, COUNT(*) AS c FROM memories GROUP BY user_id, level"
    )
    mem_by_user_level = defaultdict(lambda: defaultdict(int))
    for r in rows:
        mem_by_user_level[str(r["user_id"])][r["level"]] = int(r["c"])
    out["memories_by_user_level"] = {u: dict(lvls) for u, lvls in mem_by_user_level.items()}
    out["memories_total"] = sum(sum(lvls.values()) for lvls in mem_by_user_level.values())

    by_level = await fetch_all("SELECT level, COUNT(*) AS c FROM memories GROUP BY level")
    out["memories_by_level"] = {r["level"]: int(r["c"]) for r in by_level}

    rels = await fetch_all("SELECT relation, COUNT(*) AS c FROM memory_relations GROUP BY relation")
    out["relations_by_type"] = {r["relation"]: int(r["c"]) for r in rels}
    out["relations_total"] = sum(int(r["c"]) for r in rels)

    cs = await fetch_all("SELECT COUNT(*) AS c FROM chat_sessions")
    cm = await fetch_all("SELECT COUNT(*) AS c FROM chat_messages")
    out["chat_sessions_total"] = int(cs[0]["c"])
    out["chat_messages_total"] = int(cm[0]["c"])

    ext = await fetch_all("SELECT status, COUNT(*) AS c FROM memory_extract_logs GROUP BY status")
    out["extract_logs_by_status"] = {r["status"]: int(r["c"]) for r in ext}

    return out


def validate(logs: list[dict], metrics: dict, db: dict) -> list[ValidationResult]:
    out: list[ValidationResult] = []

    total = len(logs)
    if total == 0:
        return [ValidationResult("链路正确性", "无数据", 0, "fail", "压测日志为空")]

    ok = sum(1 for r in logs if r["ok"])
    empty = sum(1 for r in logs if r["ok"] and not r["reply"].strip())
    empty_pct = (empty / max(1, total)) * 100
    out.append(ValidationResult("链路正确性", "总轮次", total, "ok"))
    rate = ok / total * 100
    out.append(ValidationResult(
        "链路正确性", "收到 assistant 回复", f"{ok}/{total} ({rate:.1f}%)",
        "ok" if rate >= 95 else ("warn" if rate >= 80 else "fail"),
    ))
    out.append(ValidationResult(
        "链路正确性", "空回复比例", f"{empty_pct:.1f}%",
        "ok" if empty_pct < 10 else ("warn" if empty_pct < 30 else "fail"),
        "模型输出只有 <think> 内容被 strip 时会出现"
    ))

    lats = [r["latency_ms"] for r in logs if r["latency_ms"] > 0]
    if lats:
        lats_sorted = sorted(lats)
        p50 = lats_sorted[len(lats_sorted) // 2]
        p95 = lats_sorted[int(len(lats_sorted) * 0.95)]
        p99 = lats_sorted[int(len(lats_sorted) * 0.99)]
        avg = statistics.mean(lats)
        out.append(ValidationResult("召回及时性", "平均延迟(ms)", int(avg),
            "ok" if avg < 15000 else ("warn" if avg < 30000 else "fail")))
        out.append(ValidationResult("召回及时性", "P50 延迟(ms)", p50, "ok"))
        out.append(ValidationResult("召回及时性", "P95 延迟(ms)", p95,
            "ok" if p95 < 30000 else ("warn" if p95 < 60000 else "fail")))
        out.append(ValidationResult("召回及时性", "P99 延迟(ms)", p99,
            "ok" if p99 < 60000 else ("warn" if p99 < 120000 else "fail"),
            "超过 30s 通常是 LLM 慢响应或流式异常"
        ))

    by_type = defaultdict(list)
    for r in logs:
        by_type[r["conv_type"]].append(r)
    mem_l0 = db["memories_by_level"].get("L0", 0)
    mem_l1 = db["memories_by_level"].get("L1", 0)
    mem_l2 = db["memories_by_level"].get("L2", 0)

    l0_count = by_type.get("L0_preference", [])
    out.append(ValidationResult("抽取合理性", "L0_preference 轮次", len(l0_count), "ok"))
    out.append(ValidationResult("抽取合理性", "L0 记忆生成", mem_l0,
        "ok" if mem_l0 > 0 else "warn",
        f"对应 {len(l0_count)} 条 L0 偏好陈述"
    ))

    l1_count = by_type.get("L1_event", [])
    out.append(ValidationResult("抽取合理性", "L1_event 轮次", len(l1_count), "ok"))
    out.append(ValidationResult("抽取合理性", "L1 记忆生成", mem_l1,
        "ok" if mem_l1 > 0 else "warn"))

    out.append(ValidationResult("抽取合理性", "L2 摘要数", mem_l2,
        "ok" if mem_l2 > 0 else "warn",
        "摘要由 extract_and_archive 在每次会话后批量生成"))

    rel_types = db["relations_by_type"]
    contradict = rel_types.get("contradict", 0)
    extend = rel_types.get("extend", 0)
    update = rel_types.get("update", 0)
    causes = rel_types.get("causes", 0)
    out.append(ValidationResult("关系识别", "causes", causes, "ok"))
    out.append(ValidationResult("关系识别", "update", update, "ok"))
    out.append(ValidationResult("关系识别", "extend", extend, "ok"))
    out.append(ValidationResult("关系识别", "contradict", contradict,
        "ok" if contradict > 0 else "warn",
        "contradict 类对话应能产生矛盾边"))

    error_counts = Counter()
    for r in logs:
        if not r["ok"]:
            error_counts[r.get("error") or "unknown"] += 1
    out.append(ValidationResult("异常分布", "错误类型计数", dict(error_counts), "ok"))
    if not error_counts:
        out.append(ValidationResult("异常分布", "无错误轮次", total, "ok"))
    empty_ok = sum(1 for r in logs if r["ok"] and not r["reply"].strip())
    out.append(ValidationResult("异常分布", "标记 ok 但回复为空", empty_ok,
        "ok" if empty_ok == 0 else "warn",
        "模型仅输出 thinking 内容时 strip 后为空,链路正常但用户体验差"))

    # 新增: 抽取日志健康度
    ext_status = db.get("extract_logs_by_status", {})
    pending = ext_status.get("pending", 0)
    success = ext_status.get("success", 0)
    failed_ext = ext_status.get("failed", 0)
    out.append(ValidationResult("抽取日志", "pending(未完成)", pending,
        "ok" if pending == 0 else "fail",
        "pending 表示 fire-and-forget 抽取任务尚未完成; LLM 慢响应会导致任务堆积"))
    out.append(ValidationResult("抽取日志", "success", success, "ok"))
    out.append(ValidationResult("抽取日志", "failed", failed_ext,
        "ok" if failed_ext == 0 else "warn"))

    return out


def render_markdown(results: list[ValidationResult], logs: list[dict], metrics: dict,
                    db: dict, verify_path: str | None = None) -> str:
    lines: list[str] = []
    lines.append("# Echo 记忆链路压测复盘报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**压测规模**: {metrics.get('total_rounds', '?')} 轮 × {len(metrics.get('per_user', []))} 用户")
    lines.append(f"**OK 轮次**: {metrics.get('ok_rounds', '?')} / {metrics.get('total_rounds', '?')}")
    lines.append(f"**平均延迟**: {metrics.get('avg_latency_ms', 0):.1f} ms")
    lines.append("")
    lines.append("## 一、链路正确性")
    lines.append("| 指标 | 值 | 状态 | 说明 |")
    lines.append("|------|----|----|------|")
    for r in results:
        if r.section == "链路正确性":
            lines.append(f"| {r.metric} | {r.value} | {r.status} | {r.note} |")
    lines.append("")

    lines.append("## 二、召回及时性")
    lines.append("| 指标 | 值 | 状态 | 说明 |")
    lines.append("|------|----|----|------|")
    for r in results:
        if r.section == "召回及时性":
            lines.append(f"| {r.metric} | {r.value} | {r.status} | {r.note} |")
    lines.append("")

    lines.append("## 三、抽取合理性")
    lines.append("### 3.1 记忆分布")
    lines.append("```")
    lines.append(f"L0 (长期偏好/身份): {db['memories_by_level'].get('L0', 0)}")
    lines.append(f"L1 (近期事件):     {db['memories_by_level'].get('L1', 0)}")
    lines.append(f"L2 (会话摘要):     {db['memories_by_level'].get('L2', 0)}")
    lines.append(f"Total:              {db['memories_total']}")
    lines.append("```")
    lines.append("")
    lines.append("### 3.2 验证")
    lines.append("| 指标 | 值 | 状态 | 说明 |")
    lines.append("|------|----|----|------|")
    for r in results:
        if r.section == "抽取合理性":
            lines.append(f"| {r.metric} | {r.value} | {r.status} | {r.note} |")
    lines.append("")

    lines.append("## 四、关系识别")
    lines.append("```")
    for rel, cnt in db["relations_by_type"].items():
        lines.append(f"  {rel:12s}: {cnt}")
    lines.append(f"  Total: {db['relations_total']}")
    lines.append("```")
    lines.append("")
    lines.append("| 指标 | 值 | 状态 | 说明 |")
    lines.append("|------|----|----|------|")
    for r in results:
        if r.section == "关系识别":
            lines.append(f"| {r.metric} | {r.value} | {r.status} | {r.note} |")
    lines.append("")

    lines.append("## 五、异常分布")
    lines.append("| 指标 | 值 | 状态 | 说明 |")
    lines.append("|------|----|----|------|")
    for r in results:
        if r.section == "异常分布":
            lines.append(f"| {r.metric} | {r.value} | {r.status} | {r.note} |")
    lines.append("")

    lines.append("## 六、按用户明细")
    lines.append("| 用户 | 轮次 | OK | 失败 | 平均延迟 |")
    lines.append("|------|------|----|------|----------|")
    for u in metrics.get("per_user", []):
        avg = u["total_latency_ms"] / max(1, u["total_rounds"])
        lines.append(
            f"| user_idx={u['user_idx']} | {u['total_rounds']} | {u['ok_rounds']} | {u['failed_rounds']} | {avg:.0f} ms |"
        )
    lines.append("")

    lines.append("## 七、典型异常轮次")
    failed = [r for r in logs if not r["ok"] or (r["ok"] and not r["reply"].strip())]
    lines.append(f"共 {len(failed)} 条异常,展示前 10 条:")
    lines.append("")
    lines.append("| 轮次 | 类型 | 用户消息 | 错误/回复 |")
    lines.append("|------|------|---------|----------|")
    for r in failed[:10]:
        err = r.get("error") or "(空回复)"
        msg = r["user_message"][:60]
        lines.append(f"| {r['round_id']} | {r['conv_type']} | {msg} | {err[:80]} |")
    lines.append("")

    statuses = [r.status for r in results]
    fail_count = statuses.count("fail")
    warn_count = statuses.count("warn")
    ok_count = statuses.count("ok")
    lines.append("## 八、总结")
    lines.append(f"- 通过: {ok_count}")
    lines.append(f"- 警告: {warn_count}")
    lines.append(f"- 失败: {fail_count}")
    lines.append("")
    if fail_count == 0 and warn_count == 0:
        lines.append("✅ 全部验证项通过,记忆链路工作正常。")
    elif fail_count == 0:
        lines.append(f"⚠️ 有 {warn_count} 项警告,需关注但不阻塞。")
    else:
        lines.append(f"❌ 有 {fail_count} 项失败,{warn_count} 项警告,需要修复。")
    lines.append("")

    # 八.1 关键发现 (问题清单)
    lines.append("### 八.1 关键问题清单")
    issues: list[str] = []
    pending = db.get("extract_logs_by_status", {}).get("pending", 0)
    if pending > 0:
        issues.append(
            f"**抽取链路阻塞**: memory_extract_logs 中有 **{pending} 条 pending 状态**。"
            f"原因是 echo-ai 的记忆抽取采用 fire-and-forget 后台任务，"
            f"LLM 慢响应时任务长时间未完成，影响后续记忆写入。建议：\n"
            f"  - 给后台抽取加超时(2~3分钟)\n"
            f"  - 失败时回写 status=failed 而不是一直 pending\n"
            f"  - 加 worker 监控, 堆积超阈值告警"
        )
    # 端到端抽取失败
    e2e_passed = False
    if verify_path and Path(verify_path).exists():
        verify_data = json.loads(Path(verify_path).read_text(encoding="utf-8"))
        for v in verify_data:
            if "端到端" in v.get("name", "") and not v["ok"]:
                issues.append(
                    f"**端到端 L0 抽取未触发**: 直接验证显示 `memories_generated=0` 但 LLM 回复正常。"
                    f"原因可能是 (a) 后台抽取超时/失败 (b) assistant_msg 在流式后为空 "
                    f"导致 extract_and_archive 提前返回。验证: `{v['detail']}`"
                )
            if "召回验证" in v.get("name", "") and v["ok"]:
                e2e_passed = True
    # LLM 速率限制
    avg_lat = metrics.get("avg_latency_ms", 0)
    if avg_lat > 30000:
        issues.append(
            f"**LLM 速率限制**: 平均响应 {avg_lat:.0f}ms (期望 < 15s)。"
            f"本机环境 LLM 在持续并发调用下降速严重，导致 1000 轮压测不可行。"
            f"建议: (a) 接入更稳定的 LLM 供应商 (b) 加 LLM 调用本地缓存 "
            f"(c) 抽改 embedding 计算与 LLM 抽取并行执行"
        )
    # 空回复比例
    empty_pct_check = (sum(1 for r in logs if r["ok"] and not r["reply"].strip()) / max(1, len(logs))) * 100
    if empty_pct_check > 5:
        issues.append(
            f"**空回复 {empty_pct_check:.1f}%**: 部分轮次 LLM 仅输出 `<think>...</think>` 内容被 strip 后为空。"
            f"建议: 增加 small_stream 的 max_tokens 上限, 或在 reply_len=0 时重试一次(已知 _should_retry_starved)"
        )

    if not issues:
        lines.append("本次测试未发现明显问题。")
    else:
        for i, issue in enumerate(issues, 1):
            lines.append(f"{i}. {issue}")
            lines.append("")
    lines.append("")
    return "\n".join(lines)


def render_pdf(markdown_text: str, output_pdf: str) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        )
        from reportlab.lib import colors
    except ImportError:
        print("reportlab 未安装,跳过 PDF 生成")
        return

    doc = SimpleDocTemplate(
        output_pdf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Echo 记忆链路压测复盘报告",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, spaceAfter=12)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=14, spaceAfter=8)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=12, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14)
    code = ParagraphStyle("code", parent=styles["Code"], fontSize=8, leading=10, fontName="Courier")

    story = []
    in_code = False
    in_table = False
    table_data: list[list[str]] = []

    def flush_table():
        nonlocal table_data
        if not table_data:
            return
        tbl = Table(table_data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.3 * cm))
        table_data = []

    for line in markdown_text.splitlines():
        if line.startswith("```"):
            if in_code:
                story.append(Spacer(1, 0.2 * cm))
            in_code = not in_code
            continue
        if in_code:
            story.append(Paragraph(line.replace(" ", "&nbsp;") or "&nbsp;", code))
            continue
        if line.startswith("|") and not in_table:
            in_table = True
            table_data = []
        if line.startswith("|") and in_table:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            table_data.append(cells)
            continue
        if in_table and not line.startswith("|"):
            flush_table()
            in_table = False
        if line.startswith("# "):
            story.append(Paragraph(line[2:], h1))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], h2))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], h3))
        elif line.strip() == "":
            story.append(Spacer(1, 0.2 * cm))
        else:
            story.append(Paragraph(line.replace("&", "&amp;"), body))
    flush_table()
    doc.build(story)
    print(f"PDF -> {output_pdf}")


async def main_async(
    metrics_path: str, logs_path: str, md_path: str, pdf_path: str,
    verify_path: str | None = None,
) -> None:
    logs = load_logs(logs_path)
    if Path(metrics_path).exists():
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    else:
        # 压测未完成, 从 logs 推断
        ok = sum(1 for r in logs if r["ok"])
        lats = [r["latency_ms"] for r in logs if r["latency_ms"] > 0]
        avg = statistics.mean(lats) if lats else 0
        metrics = {
            "total_rounds": len(logs),
            "ok_rounds": ok,
            "failed_rounds": len(logs) - ok,
            "avg_latency_ms": avg,
            "conv_type_distribution": dict(Counter(r["conv_type"] for r in logs)),
            "per_user": [],
            "_incomplete": True,
        }
    db = await db_stats()
    results = validate(logs, metrics, db)
    md = render_markdown(results, logs, metrics, db, verify_path=verify_path)

    # 附加直接验证结果
    if verify_path and Path(verify_path).exists():
        verify_data = json.loads(Path(verify_path).read_text(encoding="utf-8"))
        md += "\n\n## 九、直接链路验证 (绕过 UI)\n\n"
        md += "由于 UI 压测受 LLM 速率限制无法跑满 1000 轮, 改用直接验证脚本针对 3 个改动逐项验证:\n\n"
        md += "| 验证项 | 状态 | 详情 |\n"
        md += "|--------|------|------|\n"
        for v in verify_data:
            status = "✅ PASS" if v["ok"] else "❌ FAIL"
            md += f"| {v['name']} | {status} | {v['detail']} |\n"
        md += "\n"

    Path(md_path).write_text(md, encoding="utf-8")
    print(f"MD   -> {md_path}")
    render_pdf(md, pdf_path)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="E:/AIWorking/workspace/report/echo/stress_metrics.json")
    parser.add_argument("--logs", default="E:/AIWorking/workspace/report/echo/stress_logs.jsonl")
    parser.add_argument("--md", default="E:/AIWorking/workspace/report/echo/echo_memory_stress_report.md")
    parser.add_argument("--pdf", default="E:/AIWorking/workspace/report/echo/echo_memory_stress_report.pdf")
    parser.add_argument("--verify", default="E:/AIWorking/workspace/report/echo/verification_results.json")
    args = parser.parse_args()
    asyncio.run(main_async(args.metrics, args.logs, args.md, args.pdf, args.verify))


if __name__ == "__main__":
    main()