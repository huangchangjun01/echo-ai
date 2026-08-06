"""场景聚合单元测试：parse_scene_extraction / aggregate / render_*"""

from __future__ import annotations

import pytest

from parsers.scene_aggregator import (
    PerSceneRender,
    PersistentSummary,
    SceneExtraction,
    _mode_setting,
    _union_dedupe,
    aggregate,
    parse_scene_extraction,
    render_persistent,
    render_scene,
)


# ===== parse_scene_extraction =====

VALID_LLM_OUTPUT = """\
- 人物: 阿强(蓝色卫衣, 背双肩包)、小美(白裙, 长发)
- 场景: 公园入口，黄昏
- 动作: 阿强挥手打招呼、小美转身看向阿强
- 物品: 双肩包(阿强背上)、水瓶(阿强手中)
- 文字: 无
- 表情: 阿强微笑、小美惊讶
- 颜色方位: 暖橙色调，人物居中，背景虚化
- 变化: 初始

黄昏的公园入口，阿强（蓝色卫衣，背着双肩包）微笑着挥手打招呼，小美（白裙，长发）转身看向他。背景为暖橙色调，人物居于画面中央，景深略虚化。
"""


def test_parse_valid_dual_output() -> None:
    e = parse_scene_extraction(VALID_LLM_OUTPUT, idx=1, source="park.mp4")
    assert e is not None
    assert e.idx == 1
    assert e.source == "park.mp4"
    assert e.characters == ["阿强(蓝色卫衣, 背双肩包)", "小美(白裙, 长发)"]
    assert e.setting == "公园入口，黄昏"
    assert e.actions == ["阿强挥手打招呼", "小美转身看向阿强"]
    assert e.objects == ["双肩包(阿强背上)", "水瓶(阿强手中)"]
    assert e.text == "无"  # 原文保留"无"字（不当作文字内容）
    assert e.expressions == ["阿强微笑", "小美惊讶"]
    assert e.color_spatial == "暖橙色调，人物居中，背景虚化"
    assert e.deltas == "初始"
    assert e.prose.startswith("黄昏的公园入口")
    assert "阿强" in e.prose


def test_parse_uses_fullwidth_colon() -> None:
    raw = """\
- 人物：A
- 场景：S
- 动作：M
- 文字：无
- 变化：初始

单句描述。"""
    e = parse_scene_extraction(raw, 1, "x.mp4")
    assert e is not None
    assert e.characters == ["A"]
    assert e.setting == "S"
    assert e.actions == ["M"]


def test_parse_without_prose_section() -> None:
    """LLM 只输出结构化字段，没空行 prose → 仍可解析，prose 为空。"""
    raw = """\
- 人物: 路人
- 场景: 街角
- 动作: 行走
- 变化: 初始"""
    e = parse_scene_extraction(raw, 1, "x.mp4")
    assert e is not None
    assert e.prose == ""
    assert e.actions == ["行走"]


def test_parse_missing_key_fields_returns_none() -> None:
    """无 -人物 / -动作 视为解析失败。"""
    raw = """\
- 场景: 街角
- 文字: 无
- 变化: 初始

描述。"""
    assert parse_scene_extraction(raw, 1, "x.mp4") is None


def test_parse_empty_string_returns_none() -> None:
    assert parse_scene_extraction("", 1, "x.mp4") is None
    assert parse_scene_extraction("   \n\n  ", 1, "x.mp4") is None


def test_parse_text_field_normalizes_wu() -> None:
    raw = """\
- 人物: A
- 文字: 无
- 变化: 初始

desc."""
    e = parse_scene_extraction(raw, 1, "x.mp4")
    assert e is not None
    assert e.text == "无"


# ===== _union_dedupe / _mode_setting =====

def test_union_dedupe_preserves_order() -> None:
    out = _union_dedupe([
        ["阿强", "小美"],
        ["阿强", "朋友"],
        ["小美", "路人"],
    ])
    assert out == ["阿强", "小美", "朋友", "路人"]


def test_union_dedupe_normalizes_whitespace() -> None:
    out = _union_dedupe([
        ["A  B", "C"],
        ["A B", "D"],
    ])
    # "A  B" 和 "A B" 归一化后相同 → 保留首次出现
    assert out == ["A  B", "C", "D"]


def test_mode_setting_picks_majority() -> None:
    assert _mode_setting(["公园", "公园", "街角"]) == "公园"


def test_mode_setting_falls_back_to_first() -> None:
    # 2 个不同，1 个场景 50% 不到 → 取首
    assert _mode_setting(["公园", "街角", "湖边"]) == "公园"


def test_mode_setting_ignores_empty() -> None:
    assert _mode_setting(["", "公园", "公园", ""]) == "公园"


def test_mode_setting_all_empty() -> None:
    assert _mode_setting(["", "", ""]) == ""


# ===== aggregate =====

def _mk_extraction(idx: int, **kwargs) -> SceneExtraction:
    """构造一个测试用的 SceneExtraction。"""
    defaults = dict(
        idx=idx,
        characters=[],
        setting="",
        actions=[],
        objects=[],
        text="",
        expressions=[],
        color_spatial="",
        deltas="初始",
        prose=f"场景 {idx} 的连贯描述。",
    )
    defaults.update(kwargs)
    return SceneExtraction(**defaults)


def test_aggregate_single_scene_no_persistent() -> None:
    e = _mk_extraction(1, characters=["A"], setting="街角", prose="一段描述。")
    p, per_list = aggregate([e], source="x.mp4")
    assert p is None  # 1 个场景不显示贯穿主体
    assert len(per_list) == 1
    assert per_list[0].idx == 1
    assert per_list[0].total == 1
    assert per_list[0].prose == "一段描述。"


def test_aggregate_two_scenes_unions_characters() -> None:
    es = [
        _mk_extraction(1, characters=["阿强", "小美"], setting="公园",
                       color_spatial="暖橙", deltas="初始", prose="P1。"),
        _mk_extraction(2, characters=["阿强", "小美", "朋友"], setting="公园",
                       color_spatial="暖橙", deltas="移到步道", prose="P2。"),
    ]
    p, per_list = aggregate(es, source="x.mp4")
    assert p is not None
    assert p.characters == ["阿强", "小美", "朋友"]
    assert p.setting == "公园"
    assert len(per_list) == 2
    # scene 1: 颜色方位保留（首场景）
    assert per_list[0].color_spatial == "暖橙"
    # scene 2: 颜色方位与首场景相同 → 省略
    assert per_list[1].color_spatial == ""


def test_aggregate_keeps_changed_color_spatial() -> None:
    es = [
        _mk_extraction(1, color_spatial="暖橙色调", deltas="初始"),
        _mk_extraction(2, color_spatial="湖面反光在左", deltas="移到湖边"),
    ]
    p, per_list = aggregate(es, source="x.mp4")
    assert per_list[0].color_spatial == "暖橙色调"
    assert per_list[1].color_spatial == "湖面反光在左"


def test_aggregate_objects_union() -> None:
    es = [
        _mk_extraction(1, objects=["双肩包(背上)", "水瓶"], deltas="初始"),
        _mk_extraction(2, objects=["双肩包(背上)"], deltas="走"),
        _mk_extraction(3, objects=["双肩包(放在长椅上)"], deltas="坐下"),
    ]
    p, per_list = aggregate(es, source="x.mp4")
    assert p is not None
    # 3 个不同描述归一化后都属于"双肩包..."，但因字段值不同，归一化对比会按"双肩包"作核心
    # 实际：_normalize 不会切括号，所以"双肩包(背上)"和"双肩包(放在长椅上)"会被去重
    # 因为归一化后是 "双肩包(背上)" vs "双肩包(放在长椅上)"，空白被去掉但括号内容保留
    # 因此两者会作为不同项
    assert any("双肩包" in o for o in p.objects)
    assert "水瓶" in p.objects


def test_aggregate_text_field_unions_visible_text_only() -> None:
    es = [
        _mk_extraction(1, text="无", prose="P1。"),
        _mk_extraction(2, text="「欢迎光临」", prose="P2。"),
        _mk_extraction(3, text="无", prose="P3。"),
    ]
    p, _ = aggregate(es, source="x.mp4")
    assert p is not None
    # "无" 被忽略，"欢迎光临" 保留（去引号会做吗？这里 raw 保留）
    assert "欢迎" in p.text


def test_aggregate_empty_extractions() -> None:
    p, per_list = aggregate([], source="x.mp4")
    assert p is None
    assert per_list == []


def test_aggregate_preserves_prose_per_scene() -> None:
    es = [
        _mk_extraction(1, prose="第一段连贯描述。", deltas="初始"),
        _mk_extraction(2, prose="第二段连贯描述。", deltas="走"),
    ]
    _, per_list = aggregate(es, source="x.mp4")
    assert per_list[0].prose == "第一段连贯描述。"
    assert per_list[1].prose == "第二段连贯描述。"


# ===== render_persistent / render_scene =====

def test_render_persistent_full() -> None:
    p = PersistentSummary(
        characters=["A", "B"],
        setting="公园，黄昏",
        objects=["双肩包"],
        text="",
    )
    out = render_persistent(p)
    assert "### 贯穿主体" in out
    assert "人物: A、B" in out
    assert "场景: 公园，黄昏" in out
    assert "物品: 双肩包" in out
    assert "文字" not in out  # text 为空时省略该行


def test_render_persistent_minimal() -> None:
    p = PersistentSummary()
    out = render_persistent(p)
    # 仅标题，没有任何数据行
    assert out.strip() == "### 贯穿主体"


def test_render_scene_with_prose() -> None:
    per = PerSceneRender(
        idx=1, total=3, source="x.mp4",
        actions=["招手"], deltas="初始",
        prose="黄昏的公园入口，阿强挥手打招呼。",
    )
    out = render_scene(per)
    assert "### 片段 1/3 [来源: x.mp4]" in out
    assert "黄昏的公园入口" in out
    # 有 prose 时不应包含字段标签
    assert "动作:" not in out
    assert "变化:" not in out


def test_render_scene_fallback_to_actions_when_no_prose() -> None:
    per = PerSceneRender(
        idx=2, total=3, source="x.mp4",
        actions=["行走", "聊天"], deltas="移到步道",
    )
    out = render_scene(per)
    assert "### 片段 2/3" in out
    # prose 为空时拼 fallback
    assert "行走" in out
    assert "聊天" in out


def test_render_scene_fallback_no_data() -> None:
    per = PerSceneRender(idx=3, total=3, source="x.mp4")
    out = render_scene(per)
    assert "### 片段 3/3" in out
    assert "无可用描述" in out
