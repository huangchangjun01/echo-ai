"""Skill: create_memory_md — 生成 {memoryId}.md。"""

from __future__ import annotations

import logging
import re

from llm.client import get_llm_client
from utils.request_context import log_exception, merge_extra

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "你是「回忆记忆整理助手」。你的任务是把用户提供的「记忆主题 + 主观描述 + 各文件解析细节」"
    "整理成一份 markdown 文档，结构严格按以下五节：\n"
    "  # {topic}\n"
    "  ## 摘要（≤200 字中文概述；包含时间/地点/人物/事件要点）\n"
    "  ## 元数据（项目符号列表：时间 / 地点 / 人物 / 情感标签 / 父Skill / 前序Skill / 后续Skill / 强度 / 来源）\n"
    "  ## 记忆细节（按文件分段，格式 `### 片段 N [来源: 文件名]`）\n"
    "  ## 记忆主观描述（用户原话，可空）\n"
    "硬性要求：\n"
    "  1. 只归纳与组织，不得新增原文不存在的内容，不得改写细节语义。\n"
    "  2. 不确定的字段（元数据中时间/地点/父Skill 等）填 `未知` 或留空，绝不编造。\n"
    "  3. 来源列表使用原始文件名列表。\n"
    "  4. 摘要不超过 200 字。\n"
    "  5. 直接返回 markdown 文本，不要加任何前后缀解释。"
)


def _build_user_prompt(
    topic: str,
    subjective_desc: str,
    details: list[dict],
) -> str:
    parts: list[str] = []
    parts.append(f"记忆主题: {topic}")
    if subjective_desc:
        parts.append(f"\n用户主观描述:\n{subjective_desc}")
    if details:
        parts.append("\n各文件解析细节:")
        for i, d in enumerate(details, 1):
            name = d.get("fileName") or f"片段{i}"
            text = (d.get("detail") or "").strip()
            parts.append(f"\n[{i}] 文件: {name}\n{text}")
    return "\n".join(parts)


_SECTION_RE = re.compile(r"^##\s", re.M)

# 推理模型（MiniMax-M3 等）常把 chain-of-thought 放在 <think>...</think> 里输出；
# 必须剥离，否则会污染 md 文档开头。
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.M)
# LLM 经常用 ```markdown ... ``` 包住整篇输出（特别是被要求"返回 markdown"时）。
_FENCE_RE = re.compile(r"^\s*```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$", re.M)


def _strip_think_and_fences(text: str) -> str:
    """清理 LLM 输出里非 markdown 的噪声：推理块 / 代码围栏 / 开头解释文字。"""
    text = _THINK_RE.sub("", text)
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1)
    # 去掉 think/fence 剥离后留下的前导空白，保证下一行就是首个非空内容
    return text.lstrip("\n\r\t ")


def _dedup_leading_title(text: str, topic: str) -> str:
    """如果 LLM 输出开头就是 `# {topic}`，去掉由 patcher 预先补上的重复标题。"""
    text = text.lstrip()
    # 仅当首行恰好等于 "# {topic}" 时删掉它（patcher 预补的）
    first_line, _, rest = text.partition("\n")
    if first_line.strip() == f"# {topic}":
        return rest.lstrip("\n")
    return text


def _validate_and_patch(md: str, topic: str, subjective_desc: str, source_files: list[str]) -> str:
    """五大节缺失时按模板补齐（保证结构稳定）。"""
    text = md.strip()
    # 先剥离 <think> / 代码围栏，避免干扰结构判定
    text = _strip_think_and_fences(text).strip()
    # 再去重：如果 LLM 自己输出了 `# {topic}`，而 patcher 也补了一份，删掉 patcher 那份
    text = _dedup_leading_title(text, topic)
    has_h1 = bool(re.match(r"^#\s", text, re.M))
    sections = set(re.findall(r"^##\s+(.+)$", text, re.M))
    if not has_h1:
        text = f"# {topic}\n\n" + text
    if "摘要" not in sections:
        text += "\n\n## 摘要\n（暂无摘要）"
    if "元数据" not in sections:
        sources = ", ".join(source_files) or "未知"
        text += (
            "\n\n## 元数据\n"
            "- 时间: 未知\n"
            "- 地点: 未知\n"
            "- 人物: 用户\n"
            "- 情感标签: []\n"
            "- 父Skill: \n"
            "- 前序Skill: \n"
            "- 后续Skill: \n"
            f"- 强度: 0.5\n- 来源: [{sources}]"
        )
    if "记忆细节" not in sections:
        text += "\n\n## 记忆细节\n（暂无细节）"
    if "记忆主观描述" not in sections:
        text += f"\n\n## 记忆主观描述\n{subjective_desc or ''}"
    return text


async def build_memory_md(
    topic: str,
    subjective_desc: str,
    details: list[dict],
    *,
    source_files: list[str] | None = None,
    use_large_model: bool = False,
) -> str:
    """生成 `{memoryId}.md` 的完整内容。"""
    prompt = _build_user_prompt(topic, subjective_desc, details)
    client = get_llm_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        if use_large_model:
            resp = await client.chat(messages, max_tokens=2048, temperature=0.5)
            text = (resp.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        else:
            text = await client.small_prefix(messages, max_tokens=2048, temperature=0.5)
        if not text:
            text = ""  # 走兜底模板
    except Exception as e:
        log_exception(
            logger,
            "build_memory_md LLM failed",
            exc=e,
            level=logging.WARNING,
            stage="skill_memory_md",
            event="llm_error",
            use_large=use_large_model,
        )
        text = ""

    sources = source_files if source_files is not None else [d.get("fileName", "") for d in details]
    final = _validate_and_patch(text, topic, subjective_desc, sources)
    logger.info(
        "build_memory_md ok",
        extra=merge_extra(
            stage="skill_memory_md",
            event="ok",
            topic=topic,
            use_large=use_large_model,
            out_len=len(final),
        ),
    )
    return final


def parse_summary(md: str) -> str:
    """从已生成的 md 抽取 `## 摘要` 段落文本（用于写 EchoRecall）。

    防御性处理：先剥离 <think> 块 / 代码围栏，避免历史脏数据里残留的噪声污染向量。
    """
    text = _strip_think_and_fences(md)
    m = re.search(r"##\s+摘要\s*\n+(.+?)(?=\n##\s|\Z)", text, re.S)
    return (m.group(1).strip() if m else "")[:1000]


def remove_source_section(md: str, file_name: str) -> str:
    """删除某文件对应的 `### 片段 N [来源: file_name]` 整段（含其内容直到下一个 ### 或 ##）。"""
    pattern = re.compile(
        rf"###\s+\S+\s+\[来源:\s*{re.escape(file_name)}\][\s\S]*?(?=\n###\s|\n##\s|$)",
        re.M,
    )
    new = pattern.sub("", md)
    # 多个连续换行折叠
    new = re.sub(r"\n{3,}", "\n\n", new).strip() + "\n"
    return new


def update_source_metadata(md: str, removed_file: str, remaining_files: list[str]) -> str:
    """从元数据 `- 来源: [a, b]` 中移除某个文件名。"""
    pattern = re.compile(r"(- 来源:\s*\[)([^\]]*)(\])")
    def _sub(m: re.Match) -> str:
        items = [x.strip() for x in m.group(2).split(",") if x.strip() and x.strip() != removed_file]
        items = items or remaining_files
        return f"{m.group(1)}{', '.join(items)}{m.group(3)}"
    return pattern.sub(_sub, md, count=1)