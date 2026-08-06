"""视频场景结构化聚合：把 N 个场景的结构化抽取结果合并为「贯穿主体 + 精简 per-scene」。

Stage 1 LLM 输出格式（双段）：
    【第一段】8 字段结构化 markdown（`- 字段: 值`）
    【第二段】1~2 句连贯中文 prose（与第一段之间用空行分隔）

Stage 2（本模块）：解析第一段 → 聚合 → 输出 (PersistentSummary, [PerSceneRender])。
Stage 3（skills/__init__.py）：用第二段 prose 直接渲染最终片段。

设计原则：
  - 人物/物品/文字：union 跨场景，仅在「贯穿主体」出现一次
  - 场景：取众数（出现 ≥ ceil(N/2) 次）
  - 动作/表情：仅 per-scene（场景特定）
  - 颜色方位：仅当与首场景不同时保留
  - 变化：永远保留
  - 解析失败 → 返回 None，调用方走旧路径（自由 prose）
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger("echo-ai.parsers")

# 8 个合法字段名（与 Stage 1 prompt 对齐）
_FIELDS = (
    "人物",
    "场景",
    "动作",
    "物品",
    "文字",
    "表情",
    "颜色方位",
    "变化",
)

# 行内字段正则：`- 字段: 值`，容忍全角冒号 `：`
_FIELD_RE = re.compile(r"^\s*-\s*(?P<key>人物|场景|动作|物品|文字|表情|颜色方位|变化)\s*[:：]\s*(?P<val>.*)$", re.M)


@dataclass(frozen=True)
class SceneExtraction:
    """Stage 1 解析后的单场景结构化数据。"""

    idx: int
    source: str = ""
    characters: list[str] = field(default_factory=list)
    setting: str = ""
    actions: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    text: str = ""
    expressions: list[str] = field(default_factory=list)
    color_spatial: str = ""
    deltas: str = "初始"
    prose: str = ""  # Stage 1 第二段：1~2 句连贯描述（用于最终渲染）


@dataclass(frozen=True)
class PersistentSummary:
    """贯穿全片的主体表（聚合 N 个场景后输出一次）。"""

    characters: list[str] = field(default_factory=list)
    setting: str = ""
    objects: list[str] = field(default_factory=list)
    text: str = ""


@dataclass(frozen=True)
class PerSceneRender:
    """单场景精简渲染。characters/objects/setting 已在 PersistentSummary 出现一次。"""

    idx: int
    total: int
    source: str
    actions: list[str] = field(default_factory=list)
    expressions: list[str] = field(default_factory=list)
    color_spatial: str = ""  # 空串 = 与首场景相同，已省略
    deltas: str = "初始"
    prose: str = ""  # Stage 1 给的 1~2 句连贯描述（直接用于渲染）


def _split_list(s: str) -> list[str]:
    """按 `、` 或顶层 `,` / `;` 切分字符串；空串返回 []。

    注意：括号内的 `,` 不切分（避免 `阿强(蓝色卫衣, 背双肩包)` 被误切）。
    """
    s = (s or "").strip()
    if not s or s in ("无", "无内容"):
        return []
    out: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch in "（(":
            depth += 1
            buf.append(ch)
        elif ch in "）)":
            depth -= 1
            buf.append(ch)
        elif ch in "、,;；" and depth == 0:
            part = "".join(buf).strip()
            if part:
                out.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def _normalize(s: str) -> str:
    """归一化用于去重比较。"""
    return re.sub(r"\s+", "", s or "")


def parse_scene_extraction(
    raw: str,
    idx: int,
    source: str,
) -> SceneExtraction | None:
    """解析 Stage 1 LLM 输出。

    格式预期：
      - 字段: 值
      - 字段: 值
      ...

      <空行>

      <1~2 句连贯 prose>

    解析失败（缺关键字段、无 `- 字段:` 行）→ 返回 None。
    """
    if not raw or not raw.strip():
        return None

    # 切分第一段（结构化字段）与第二段（prose）
    # 用连续两个换行作为分隔
    parts = re.split(r"\n\s*\n", raw.strip(), maxsplit=1)
    fields_block = parts[0]
    prose = parts[1].strip() if len(parts) > 1 else ""

    # 解析字段
    field_map: dict[str, str] = {}
    for m in _FIELD_RE.finditer(fields_block):
        field_map[m.group("key")] = m.group("val").strip()

    # 关键字段缺失则视为解析失败
    if "人物" not in field_map and "动作" not in field_map:
        return None

    return SceneExtraction(
        idx=idx,
        source=source,
        characters=_split_list(field_map.get("人物", "")),
        setting=(field_map.get("场景") or "").strip(),
        actions=_split_list(field_map.get("动作", "")),
        objects=_split_list(field_map.get("物品", "")),
        text=(field_map.get("文字") or "").strip(),
        expressions=_split_list(field_map.get("表情", "")),
        color_spatial=(field_map.get("颜色方位") or "").strip(),
        deltas=(field_map.get("变化") or "初始").strip(),
        prose=prose,
    )


def _union_dedupe(items_list: Iterable[list[str]]) -> list[str]:
    """把多组 list 合并去重（按归一化字符串比较），保持首次出现顺序。"""
    seen: set[str] = set()
    out: list[str] = []
    for items in items_list:
        for x in items:
            n = _normalize(x)
            if not n or n in seen:
                continue
            seen.add(n)
            out.append(x)
    return out


def _mode_setting(settings: list[str]) -> str:
    """取众数（出现 ≥ ceil(N/2) 次），否则取首个非空。"""
    non_empty = [s for s in settings if s and s.strip()]
    if not non_empty:
        return ""
    counter = Counter(non_empty)
    n = len(non_empty)
    threshold = (n + 1) // 2  # ceil(n/2)
    most_common, count = counter.most_common(1)[0]
    if count >= threshold:
        return most_common
    return non_empty[0]


def aggregate(
    extractions: list[SceneExtraction],
    source: str = "",
) -> tuple[PersistentSummary | None, list[PerSceneRender]]:
    """聚合 N 个场景的结构化数据。

    返回 (PersistentSummary, [PerSceneRender])。
    1 个场景 → (None, [...])  # 不显示贯穿主体
    ≥2 场景 → (PersistentSummary, [...])

    `source` 来自调用方（视频文件名），所有 per-scene 共享。
    若 extractions[0].source 非空则优先用之。
    """
    if not extractions:
        return None, []

    total = len(extractions)
    # source 优先用第一个场景的，否则用入参
    final_source = extractions[0].source or source

    # 1 个场景：不显示贯穿主体
    if total == 1:
        e = extractions[0]
        per = PerSceneRender(
            idx=e.idx,
            total=total,
            source=final_source,
            actions=e.actions,
            expressions=e.expressions,
            color_spatial=e.color_spatial,
            deltas=e.deltas or "初始",
            prose=e.prose,
        )
        return None, [per]

    # ≥2 场景：union
    persistent = PersistentSummary(
        characters=_union_dedupe(e.characters for e in extractions),
        setting=_mode_setting([e.setting for e in extractions]),
        objects=_union_dedupe(e.objects for e in extractions),
        text="、".join(
            dict.fromkeys(  # 去重保序
                e.text.strip() for e in extractions if e.text and e.text.strip() not in ("", "无")
            )
        ),
    )

    # per-scene：保留首场景的 color_spatial 作基线；与基线相同的才省略
    first_color = extractions[0].color_spatial
    per_list: list[PerSceneRender] = []
    for i, e in enumerate(extractions):
        # 首场景保留；后续场景与基线相同则省略；不同则保留
        if i == 0:
            color = e.color_spatial
        else:
            color = "" if (e.color_spatial and e.color_spatial == first_color) else e.color_spatial
        per_list.append(
            PerSceneRender(
                idx=e.idx,
                total=total,
                source=final_source,
                actions=e.actions,
                expressions=e.expressions,
                color_spatial=color,
                deltas=e.deltas or "初始",
                prose=e.prose,
            )
        )

    logger.info(
        "scene_aggregator: aggregate ok",
        extra={
            "stage": "scene_aggregator",
            "event": "aggregate_ok",
            "scenes": total,
            "characters": len(persistent.characters),
            "objects": len(persistent.objects),
        },
    )
    return persistent, per_list


def render_persistent(p: PersistentSummary) -> str:
    """渲染 ### 贯穿主体 markdown 段。"""
    lines = ["### 贯穿主体"]
    if p.characters:
        lines.append(f"- 人物: {'、'.join(p.characters)}")
    if p.setting:
        lines.append(f"- 场景: {p.setting}")
    if p.objects:
        lines.append(f"- 物品: {'、'.join(p.objects)}")
    if p.text:
        lines.append(f"- 文字: {p.text}")
    return "\n".join(lines)


def render_scene(per: PerSceneRender) -> str:
    """渲染单场景 `### 片段 i/N [来源: ...]` + 1~2 句连贯 prose。

    优先使用 Stage 1 输出的 `prose` 字段（自然流畅）。
    若 prose 为空（解析失败 fallback）→ 不渲染空段。
    """
    head = f"### 片段 {per.idx}/{per.total} [来源: {per.source}]"
    if per.prose:
        return f"{head}\n\n{per.prose}"
    # fallback：prose 为空时拼一个简短版（不应出现，但兜底不丢场景）
    bits: list[str] = []
    if per.actions:
        bits.append("，".join(per.actions) + "。")
    if per.color_spatial:
        bits.append(per.color_spatial + "。")
    if per.deltas and per.deltas != "初始":
        bits.append(per.deltas + "。")
    body = "".join(bits) or "（无可用描述）"
    return f"{head}\n\n{body}"
