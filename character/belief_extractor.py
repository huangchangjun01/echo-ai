"""从对话中自动抽取新的「角色信念」候选(M3 LLM 抽取实现)。

设计要点:
- 输入:最近 N 条 user / assistant 消息(默认 10 轮,共 20 条)
- 输出:结构化信念列表 [{topic, stance, confidence, evidence_quote}]
- 走 LLM(大模型)做"对话观察者"角色,判断哪些新立场需要被角色记住
- 抽取结果不直接落主表,而是创建 role_evolution_suggestion 提案,等用户 Apply
- LLM 不可用时 graceful 跳过(返回空列表),不阻塞主流程

M7 Feedback 触发链也会复用本函数(同 M3 抽取能力,不同触发条件)。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from utils.request_context import log_exception

logger = logging.getLogger(__name__)

# LLM prompt:观察最近对话,提取新立场
BELIEF_EXTRACTION_PROMPT = """你是「角色观察者」助手,任务是分析最近的对话,判断「角色」对哪些话题形成了新的立场或显著修正。

## 任务
阅读下列对话,提取 0~3 条**新出现或显著改变**的信念。每条输出包含:
- topic: 立场主题,10~30 字,简洁
- stance: 角色的具体立场,1~2 句话
- confidence: 角色对该立场的把握度 [0, 1]
- evidence_quote: 触发该立场形成的最关键 1 句话(≤60 字)

## 规则
- 只输出**新出现**或**显著修正**的立场,不要重复对话中已显式给出的常识
- 立场应反映「角色观点」,而非客观事实
- confidence < 0.5 的不要提取(把握度太低不值得形成信念)
- 严格按 JSON 数组格式输出,无新立场时输出 []

## 输出格式(严格遵守)
[
  {{"topic": "...", "stance": "...", "confidence": 0.8, "evidence_quote": "..."}}
]

## 对话上下文
{history}

## 角色已有信念(避免重复)
{existing_beliefs}
"""


async def extract_beliefs_from_conversation(
    user_id: str,
    role_id: str = "default",
    *,
    recent_n: int = 10,
    existing_beliefs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """从最近对话中抽取角色新信念。

    Args:
        user_id: 用户 ID
        role_id: 角色 ID
        recent_n: 取最近 N 轮对话(每轮 user+assistant 共 2 条)
        existing_beliefs: 角色已有信念列表(用于 prompt 避免重复提取)

    Returns:
        list[dict]:每条 {topic, stance, confidence, evidence_quote}
        失败/LLM 不可用 → 返回空列表
    """
    if not user_id:
        return []

    try:
        from database import fetch_all
        from llm import get_llm_client

        # 1. 加载最近 N 轮对话
        rows = await fetch_all(
            """
            SELECT role, content, created_at
            FROM chat_messages
            WHERE user_id=%s AND role_id=%s AND role IN ('user', 'assistant')
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (user_id, role_id, recent_n * 2),
        )
        if not rows or len(rows) < 2:
            logger.debug(
                "belief extract skip: not enough messages",
                extra={"stage": "character.belief_extract", "user_id": user_id},
            )
            return []
        # 按时间正序拼成 history
        history_lines: list[str] = []
        for r in reversed(list(rows)):
            role_zh = "用户" if r.get("role") == "user" else "角色"
            content = (r.get("content") or "").strip()
            if content:
                history_lines.append(f"{role_zh}:{content[:200]}")
        history = "\n".join(history_lines)
        if not history.strip():
            return []

        # 2. 拼 prompt(已有信念列表压缩成单行,避免 prompt 过长)
        existing_summary = ""
        if existing_beliefs:
            topics = [b.get("topic", "") for b in existing_beliefs if b.get("topic")]
            if topics:
                existing_summary = "已有信念主题:" + "、".join(topics[:10])

        prompt = BELIEF_EXTRACTION_PROMPT.format(
            history=history,
            existing_beliefs=existing_summary or "(无)",
        )

        # 3. 调大模型(同步 chat,带超时,失败兜底)
        client = get_llm_client()
        resp = await client.chat(
            messages=[
                {"role": "system", "content": "你是结构化抽取助手,只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        text = _extract_text(resp)
        if not text:
            return []

        # 4. 解析 JSON 响应(兼容 ```json 包裹)
        text = text.strip()
        if text.startswith("```"):
            # 去掉 markdown 围栏
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        items = json.loads(text)
        if not isinstance(items, list):
            logger.warning(
                "belief extract: LLM returned non-list",
                extra={"stage": "character.belief_extract", "user_id": user_id},
            )
            return []

        # 5. 过滤低 confidence 和字段缺失
        out: list[dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            topic = (it.get("topic") or "").strip()
            stance = (it.get("stance") or "").strip()
            if not topic or not stance:
                continue
            try:
                conf = float(it.get("confidence") or 0)
            except (TypeError, ValueError):
                conf = 0
            if conf < 0.5:
                continue
            out.append(
                {
                    "topic": topic[:128],
                    "stance": stance[:4000],
                    "confidence": min(1.0, max(0.0, conf)),
                    "evidence_quote": (it.get("evidence_quote") or "").strip()[:500],
                }
            )
        logger.info(
            "belief extract ok",
            extra={
                "stage": "character.belief_extract",
                "event": "ok",
                "user_id": user_id,
                "role_id": role_id,
                "extracted": len(out),
            },
        )
        return out
    except Exception as e:
        log_exception(
            logger,
            "extract_beliefs_from_conversation failed",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.belief_extract",
            event="failed",
            user_id=user_id,
            role_id=role_id,
        )
        return []


def _extract_text(resp: Any) -> str:
    """从 LLMClient.chat() 的返回 dict 中提取 text。"""
    try:
        choices = resp.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return msg.get("content") or ""
    except Exception:
        return ""


async def create_suggestions_from_extracted(
    user_id: str,
    role_id: str,
    extracted: list[dict[str, Any]],
    *,
    source: str = "memory_extract",
) -> int:
    """把抽取结果通过内部 API 写入 role_evolution_suggestion。

    Returns:
        成功创建的 suggestion 数
    """
    if not extracted:
        return 0
    try:
        from config.config import get_settings
        from remote.echo_core_client import get_echo_core_client

        client = get_echo_core_client()
        count = 0
        for it in extracted:
            payload = {
                "topic": it["topic"],
                "stance": it["stance"],
                "confidence": it["confidence"],
                "source": source,
            }
            suggestion_json = json.dumps(payload, ensure_ascii=False)
            r = await client.create_suggestion(
                user_id=user_id,
                role_id=role_id,
                target_type="belief",
                suggestion_json=suggestion_json,
                source=source,
                confidence=it["confidence"],
                reason=f"对话抽取:{it.get('evidence_quote', '')[:80]}",
            )
            if r:
                count += 1
        return count
    except Exception as e:
        log_exception(
            logger,
            "create_suggestions_from_extracted failed",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.belief_extract",
            event="create_failed",
            user_id=user_id,
        )
        return 0