"""Skill: create_memory_md — 生成 {memoryId}.md。"""

from __future__ import annotations

import logging
import re

from llm.client import get_llm_client
from utils.request_context import log_exception, merge_extra

logger = logging.getLogger(__name__)


_ABSTRACT_SYSTEM = (
    "你是「回忆记忆整理助手」。根据用户提供的「记忆主题 + 主观描述」生成两个 markdown 节。\n"
    "硬性要求：\n"
    "  1. ## 摘要（≤200 字中文概述；包含时间/地点/人物/事件要点）\n"
    "  2. ## 元数据（项目符号列表：时间 / 地点 / 人物 / 情感标签 / 强度 / 来源；未知填「未知」）\n"
    "  3. 只归纳用户给的信息，禁止编造新内容\n"
    "  4. 来源列表用原始文件名\n"
    "  5. 直接返回两节 markdown，不要任何前后缀解释"
)

_SEGMENT_SYSTEM = (
    "你是「回忆记忆整理助手」。根据用户提供的「单个文件描述」生成一个 markdown 子段。\n"
    "硬性要求：\n"
    "  1. 必须以 `### 片段 {idx}/{total} [来源: {file_name}]` 开头（花括号是占位符）\n"
    "  2. 200~500 字中文\n"
    "  3. 保留所有可见细节（人物/动作/场景/物品/文字/表情/颜色/方位）\n"
    "  4. 不压缩、不总结、不归纳；宁多写不少写\n"
    "  5. 只基于提供的描述，禁止编造新内容\n"
    "  6. 直接返回 markdown 子段，不要任何前后缀解释"
)


_SECTION_RE = re.compile(r"^##\s", re.M)

REASONING_TAG_OPEN = chr(60) + 'think' + chr(62)
REASONING_TAG_CLOSE = chr(60) + '/think' + chr(62)
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.M)
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
    # 先剥离  (html]> / 代码围栏，避免干扰结构判定
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
            "- 强度: 0.5\n"
            f"- 来源: [{sources}]"
        )
    if "记忆细节" not in sections:
        text += "\n\n## 记忆细节\n（暂无细节）"
    if "记忆主观描述" not in sections:
        text += f"\n\n## 记忆主观描述\n{subjective_desc or ''}"
    return text


async def _generate_abstract_section(
    client,
    topic: str,
    subjective_desc: str,
    use_large: bool,
) -> str:
    """build_memory_md 阶段 1：生成 ## 摘要 + ## 元数据 两节。

    仅 topic + subjective_desc 入 prompt，输出 ≤ 500 tokens，
    远小于 MiniMax-M3 上下文窗口。

    失败兜底：返回空串，由下游 _validate_and_patch 补齐占位段。
    """
    if not topic and not subjective_desc:
        return ""
    prompt = (
        f"记忆主题: {topic or '未命名'}\n"
        f"用户主观描述:\n{subjective_desc or '（无）'}\n\n"
        "请按上述系统硬性要求，输出 ## 摘要 和 ## 元数据 两节。\n"
        "重要：直接输出 markdown 内容，不要在  (html]> 里反复推演——本题信息已经齐备。"
    )
    messages = [
        {"role": "system", "content": _ABSTRACT_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    try:
        if use_large:
            resp = await client.chat(messages, max_tokens=512, temperature=0.5)
            text = (resp.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        else:
            text = await client.small_prefix(messages, max_tokens=512, temperature=0.5)
        if text:
            # 二次剥离 think/代码围栏
            text_clean = _strip_think_and_fences(text).strip()
            # 智能兜底：剥离后如果实际正文极短，说明 LLM 输出大部分是 think。
            # 这里不强制覆盖——因为 abstract 阶段只有 topic + subjective 可用，
            # 真无内容时下游 _split_abstract_into_sections 会基于 subjective_desc
            # 合成默认摘要。但低于阈值时打印 warning，便于运维观测。
            if len(text_clean) < _MIN_ABSTRACT_CONTENT_CHARS:
                logger.warning(
                    "build_memory_md abstract too short, downstream will fall back",
                    extra=merge_extra(
                        stage="skill_memory_md",
                        event="abstract_too_short",
                        use_large=use_large,
                        clean_len=len(text_clean),
                        threshold=_MIN_ABSTRACT_CONTENT_CHARS,
                    ),
                )
            text = text_clean or text
            logger.info(
                "build_memory_md abstract ok",
                extra=merge_extra(
                    stage="skill_memory_md",
                    event="abstract_ok",
                    use_large=use_large,
                    out_len=len(text),
                ),
            )
        return text
    except Exception as e:
        log_exception(
            logger,
            "build_memory_md abstract failed",
            exc=e,
            level=logging.WARNING,
            stage="skill_memory_md",
            event="abstract_error",
            use_large=use_large,
        )
        return ""


def _render_video_sections(
    scenes_structured: list[dict],
    file_name: str,
) -> str:
    """把视频 N 个场景的结构化数据渲染成"贯穿主体 + N 个 ### 片段"。

    直接消费 scene_aggregator 的 parse + aggregate + render，不调 LLM。
    Plan B 的核心：避免 LLM 把短 prose 重新展开成冗长描述。
    """
    from parsers.scene_aggregator import (
        PerSceneRender,
        PersistentSummary,
        SceneExtraction,
        aggregate,
        render_persistent,
        render_scene,
    )

    # 1) 重建 SceneExtraction 列表（detail.meta → dataclass）
    extractions: list[SceneExtraction] = []
    for s in scenes_structured:
        extractions.append(
            SceneExtraction(
                idx=s.get("idx", len(extractions) + 1),
                source=file_name,
                characters=s.get("characters", []) or [],
                setting=s.get("setting", "") or "",
                actions=s.get("actions", []) or [],
                objects=s.get("objects", []) or [],
                text=s.get("text", "") or "",
                expressions=s.get("expressions", []) or [],
                color_spatial=s.get("color_spatial", "") or "",
                deltas=s.get("deltas", "初始") or "初始",
                prose=s.get("prose", "") or "",
            )
        )

    # 2) 聚合
    persistent, per_list = aggregate(extractions, source=file_name)

    # 3) 渲染
    parts: list[str] = []
    if persistent is not None:
        parts.append(render_persistent(persistent))
    for per in per_list:
        parts.append(render_scene(per))

    return "\n\n".join(parts)


async def _generate_segment_section(
    client,
    detail: dict,
    idx: int,
    total: int,
    use_large: bool,
) -> str:
    """build_memory_md 阶段 2：为单个 detail 生成一个 ### 片段 子段。

    视频结构化路径：当 detail.meta.video_scenes_structured 存在时，
    直接用 scene_aggregator 渲染（贯穿主体 + 每场景 prose），不走 LLM 润色。
    这是为 Plan B 设计的"零 LLM 重写"路径——避免 LLM 把短 prose 重新展开成冗长
    描述（之前反馈的"重复描述人物外貌"症状的根因）。

    失败兜底：直接用原文 detail 拼成"### 片段{i}/{n} + 内容"。
    这是关键防御——LLM 永不丢内容。
    """
    file_name = (detail.get("fileName") or f"片段{idx}").strip()
    detail_text = (detail.get("detail") or "").strip()
    if not detail_text:
        return ""

    # Plan B 路径：视频结构化场景 → 直接渲染，不再 LLM 润色
    meta = detail.get("meta") or {}
    scenes_structured = meta.get("video_scenes_structured")
    if scenes_structured and any(s.get("structured") for s in scenes_structured):
        return _render_video_sections(scenes_structured, file_name)

    # 旧路径：LLM 润色（图片/音频/文本/未结构化视频）
    system_prompt = _SEGMENT_SYSTEM.format(idx=idx, total=total, file_name=file_name)
    user_prompt = (
        f"片段序号: {idx}/{total}\n"
        f"来源文件: {file_name}\n"
        f"原始解析细节（描述）:\n{detail_text}\n\n"
        "请按系统要求输出一个 ### 片段 子段。\n"
        "重要：直接输出 markdown 内容，不要在  (html]> 里反复推演——本题内容已经齐备，你的工作只是格式化。"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        if use_large:
            resp = await client.chat(messages, max_tokens=1200, temperature=0.3)
            text = (resp.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        else:
            text = await client.small_prefix(messages, max_tokens=1200, temperature=0.3)
        if text:
            # 二次防御：剥离 think 块 / 代码围栏
            text = _strip_think_and_fences(text)
            # 智能兜底：剥离 think 后如果实际正文太短（说明 LLM 在 think 里花了太多 token），
            # 改用原文 detail_text，保证细节零丢失（详细阈值见 _MIN_SEGMENT_CONTENT_CHARS）。
            text_stripped = text.strip()
            if len(text_stripped) < _MIN_SEGMENT_CONTENT_CHARS and detail_text:
                logger.warning(
                    "build_memory_md segment too short, fallback to raw detail",
                    extra=merge_extra(
                        stage="skill_memory_md",
                        event="segment_too_short_fallback",
                        idx=idx, file_name=file_name,
                        llm_out_len=len(text_stripped),
                        fallback_len=len(detail_text),
                    ),
                )
                text = f"### 片段{idx}/{total} [来源: {file_name}]\n\n{detail_text}"
            else:
                if not text.startswith("#"):
                    # 缺标题：补回标准头部
                    text = f"### 片段{idx}/{total} [来源: {file_name}]\n\n{text}"
            logger.info(
                "build_memory_md segment ok",
                extra=merge_extra(
                    stage="skill_memory_md",
                    event="segment_ok",
                    idx=idx, total=total, file_name=file_name,
                    use_large=use_large,
                    out_len=len(text),
                ),
            )
            return text
        # 拿到空串：兜底用原文
        logger.warning(
            "build_memory_md segment empty, fallback to raw detail",
            extra=merge_extra(
                stage="skill_memory_md",
                event="segment_fallback",
                idx=idx, file_name=file_name,
            ),
        )
    except Exception as e:
        log_exception(
            logger,
            "build_memory_md segment failed",
            exc=e,
            level=logging.WARNING,
            stage="skill_memory_md",
            event="segment_error",
            idx=idx, file_name=file_name,
        )
    # 兜底：LLM 失败 / 返回空 都回退到原文 detail（细节零丢失）
    return f"### 片段{idx}/{total} [来源: {file_name}]\n\n{detail_text}"


async def build_memory_md(
    topic: str,
    subjective_desc: str,
    details: list[dict],
    *,
    source_files: list[str] | None = None,
    use_large_model: bool = False,
) -> str:
    """生成 `{memoryId}.md` 的完整内容（分段多次调用版本，详见模块顶部注释）。

    入口参数与旧版本保持一致（向后兼容），内部结构改为：
      - 阶段 1：摘要 + 元数据（仅依赖 topic/subjective_desc）
      - 阶段 2：每个 detail 独立过 LLM，单段 prompt 远低于 MiniMax-M3 窗口
      - 末端：字符串拼接，不调 LLM；_validate_and_patch 兜底缺失节

    抗 LLM 输出退化说明（2026-08 用户复测反馈）：
      MiniMax-M3 是推理模型，会输出大量  (html]>...  (html]> 块再写正文。
      某些情况下 LLM 会陷入"思考过多、压缩正文"的状态——比如原始
      LLM 输出 2521 字符，剥离 think 后只剩 30 字符，远远不够描述
      视频内容。下面的阈值 + 智能兜底专门防这种情况。
"""

_MIN_SEGMENT_CONTENT_CHARS = 200  # 阶段 2 剥离 think 后实际正文 < 此值，改用原文 detail 兜底
_MIN_ABSTRACT_CONTENT_CHARS = 80  # 阶段 1 摘要最低字数阈值


async def build_memory_md(
    topic: str,
    subjective_desc: str,
    details: list[dict],
    *,
    source_files: list[str] | None = None,
    use_large_model: bool = False,
) -> str:
    """生成 `{memoryId}.md` 的完整内容（分段多次调用版本，详见模块顶部注释）。

    入口参数与旧版本保持一致（向后兼容），内部结构改为：
      - 阶段 1：摘要 + 元数据（仅依赖 topic/subjective_desc）
      - 阶段 2：每个 detail 独立过 LLM，单段 prompt 远低于 MiniMax-M3 窗口
      - 末端：字符串拼接，不调 LLM；_validate_and_patch 兜底缺失节

    收益：
      - 不会触发 MiniMax-M3 上下文窗口限制（解决用户反馈的根因）
      - 每个 detail 独立过 LLM，细节 100% 保留（不丢任何视觉信息）
      - 任一阶段 LLM 失败都有兜底，绝不输出空文件
    """
    sources = source_files if source_files is not None else [d.get("fileName", "") for d in details]
    client = get_llm_client()
    use_large = bool(use_large_model)

    # 阶段 1：摘要 + 元数据
    abstract_raw = await _generate_abstract_section(client, topic, subjective_desc, use_large)
    abstract_clean = _strip_think_and_fences(abstract_raw) if abstract_raw else ""

    # 阶段 2：每个 detail 一个片段
    segments: list[str] = []
    total = len(details)
    for i, d in enumerate(details):
        seg = await _generate_segment_section(client, d, i + 1, total, use_large)
        if seg:
            segments.append(seg)

    # ===== 拆分 abstract 为 摘要 / 元数据 =====
    abstract_part, metadata_part = _split_abstract_into_sections(abstract_clean, topic, subjective_desc, sources)

    # ===== 按规范 5 节顺序组装（不管 abstract_clean 是否为空）=====
    nl = chr(10) + chr(10)  # "## xxx\n\n" 中间那两换行
    parts: list[str] = [f"# {topic}"]
    parts.append("## 摘要" + nl + abstract_part)
    parts.append("## 元数据" + nl + metadata_part)
    if segments:
        # 修正：原 nl.replace(chr(10), "").join(segments) 把所有片段无缝拼接成一坨，
        # 多个 `### 片段 N/M` 会黏在一起，用户看不到分段。改为 "\n\n" 分隔。
        parts.append("## 记忆细节" + nl + "\n\n".join(segments))
    else:
        parts.append("## 记忆细节" + nl + "（暂无细节）")
    if subjective_desc:
        parts.append(f"## 记忆主观描述" + nl + subjective_desc)
    else:
        parts.append("## 记忆主观描述" + nl + "（无）")

    md = nl.join(parts)
    final = _validate_and_patch(md, topic, subjective_desc, sources)

    logger.info(
        "build_memory_md ok",
        extra=merge_extra(
            stage="skill_memory_md",
            event="ok",
            topic=topic,
            use_large=use_large,
            segments=len(segments),
            abstract_len=len(abstract_clean),
            out_len=len(final),
        ),
    )
    return final


def _split_abstract_into_sections(
    abstract_text: str, topic: str, subjective_desc: str, sources: list[str]
) -> tuple[str, str]:
    """把阶段 1 的 LLM 输出拆成 (摘要内容, 元数据内容)。

    - 若同时有 ## 摘要 和 ## 元数据，按 LLM 原文拆分
    - 若只有其中一个，只返回该部分，另一部分走默认合成
    - 若 LLM 输出完全为空（或剥离 think 后为空），基于 subjective_desc
      + topic 合成 default 摘要，保证"## 摘要" 永远有内容
    """
    if not abstract_text.strip():
        # 兜底：基于 topic + subjective_desc 合成简短摘要
        summary = _synthesize_default_summary(topic, subjective_desc)
        return summary, _synthesize_default_metadata(subjective_desc, sources)

    # 检测 ## 摘要 / ## 元数据 段落（一行匹配 + lazy 跨段捕获）
    a_m = re.search(
        r"##\s*摘要[^\n]*\n+(.+?)(?=\n##\s|\Z)",
        abstract_text,
        re.S,
    )
    md_m = re.search(
        r"##\s*元数据[^\n]*\n+(.+?)(?=\n##\s|\Z)",
        abstract_text,
        re.S,
    )

    a_part = (a_m.group(1).strip() if a_m else abstract_text.strip()) or _synthesize_default_summary(topic, subjective_desc)
    m_part = (md_m.group(1).strip() if md_m else "") or _synthesize_default_metadata(subjective_desc, sources)
    return a_part, m_part


def _synthesize_default_summary(topic: str, subjective_desc: str) -> str:
    """若 LLM 阶段 1 没产出摘要，从 topic + subjective_desc 拼一个简易摘要。"""
    _nl = chr(10)
    bits = []
    if topic:
        bits.append(f"主题: {topic}")
    if subjective_desc:
        bits.append(subjective_desc.strip().split(_nl)[0][:200])
    return (_nl.join(bits) if bits else "（暂无摘要）").strip()


def _synthesize_default_metadata(subjective_desc: str, sources: list[str]) -> str:
    """元数据默认模板。"""
    _nl = chr(10)
    src_list = ", ".join(sources) if sources else "未知"
    return (
        f"- 时间: 未知{_nl}"
        f"- 地点: 未知{_nl}"
        f"- 人物: 用户{_nl}"
        f"- 情感标签: []{_nl}"
        f"- 强度: 0.5{_nl}"
        f"- 来源: [{src_list}]"
    )


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