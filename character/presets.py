"""6 种预设 OCEAN 五维基线值（PRD 02_PRD_data_model.md）。

设计要点：
- 五维 ∈ [-1, 1] 连续值
- 6 种预设覆盖「陪伴/理性/元气/温柔/毒舌/咨询」6 种典型人格底色
- 冷启动时根据用户选择的 presetType 直接落到 role_traits 表（首版）
"""

from __future__ import annotations

from typing import TypedDict


class OceanTraits(TypedDict):
    openness: float          # O: 开放性
    conscientiousness: float # C: 尽责性
    extraversion: float      # E: 外向性
    agreeableness: float     # A: 宜人性
    neuroticism: float       # N: 神经质


# 6 种预设的 OCEAN 基线（PRD 02_PRD_data_model.md）
OCEAN_PRESETS: dict[str, OceanTraits] = {
    "companion_default": OceanTraits(
        openness=0.4, conscientiousness=-0.1, extraversion=0.2, agreeableness=0.7, neuroticism=-0.4,
    ),
    "rational_advisor": OceanTraits(
        openness=0.2, conscientiousness=0.7, extraversion=-0.3, agreeableness=0.2, neuroticism=-0.5,
    ),
    "energetic_girl": OceanTraits(
        openness=0.6, conscientiousness=-0.3, extraversion=0.8, agreeableness=0.6, neuroticism=-0.6,
    ),
    "gentle_senior": OceanTraits(
        openness=0.3, conscientiousness=0.4, extraversion=0.1, agreeableness=0.6, neuroticism=-0.3,
    ),
    "sarcastic_friend": OceanTraits(
        openness=0.5, conscientiousness=-0.2, extraversion=0.4, agreeableness=-0.2, neuroticism=-0.1,
    ),
    "psych_counselor": OceanTraits(
        openness=0.7, conscientiousness=0.6, extraversion=0.0, agreeableness=0.9, neuroticism=-0.7,
    ),
}


def get_preset_traits(preset_type: str) -> OceanTraits:
    """取预设 OCEAN 五维基线；未匹配则返回 companion_default。

    Args:
        preset_type: 6 种 presetType 之一

    Returns:
        OceanTraits 五维字典（深拷贝，避免被修改原字典）
    """
    base = OCEAN_PRESETS.get(preset_type)
    if base is None:
        return OceanTraits(**OCEAN_PRESETS["companion_default"])
    return OceanTraits(**base)


def get_preset_name() -> list[str]:
    """列出所有支持的 preset_type。"""
    return list(OCEAN_PRESETS.keys())