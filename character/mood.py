"""三桶心情系统(PRD 03_PRD_change_engine.md)。

设计要点：
- 三桶：instant(30s 半衰期) / short(30min 半衰期) / baseline(7d 半衰期)
- 耦合公式：
    new_instant = clamp(
        baseline.value * 0.20
        + short.value * 0.30
        + event.impact * event.intensity
        + personality_bias  # -0.15*N + 0.10*E + ...
        + relationship_influence  # intimacy/trust 影响
        + noise(σ=0.02)
    )
    short 衰减:   short = 0.7*short + 0.3*instant     (每 30min)
    baseline 衰减: baseline = 0.7*baseline + 0.3*short (每 24h)
- emotion 标签：happy / sad / angry / excited / neutral / concerned / thinking 等
- 表达映射(二维 valence × arousal → expression)：
    valence > 0.5, arousal high → excited
    valence > 0.5              → happy / smile
    valence ∈ [-0.5, 0.5]      → neutral
    valence < -0.5, arousal low → sad
    valence < -0.5              → concerned / cry
- mood → tone (注入到 system prompt 末段)
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Literal

# 表达式枚举(前端 PixelCharacter 8 表情)
Expression = Literal[
    "smile", "happy", "excited", "sad", "cry", "angry", "concern", "thinking", "neutral"
]

# 情绪标签候选
EmotionLabel = Literal[
    "happy", "sad", "angry", "excited", "neutral", "concerned", "thinking", "disappointed"
]

logger = logging.getLogger(__name__)


async def load_mood(user_id: str, role_id: str = "default") -> MoodSnapshot | None:
    """从 role_mood 表加载三桶心情快照。

    写入路径在 echo-core/handlers 的 UpdateMood。
    返回 None 时由 prompts_inject.fallback 走中性默认。
    """
    if not user_id:
        return None
    try:
        from database import fetch_one
        from utils.request_context import log_exception

        row = await fetch_one(
            """
            SELECT instant_val, instant_intensity, instant_emotion,
                   short_val, short_intensity, short_emotion,
                   baseline_val, baseline_intensity, baseline_emotion
            FROM role_mood
            WHERE user_id=%s AND role_id=%s
            """,
            (user_id, role_id),
        )
        if not row:
            return None
        return MoodSnapshot(
            instant_val=float(row.get("instant_val") or 0),
            instant_intensity=float(row.get("instant_intensity") or 0),
            instant_emotion=row.get("instant_emotion") or "neutral",
            short_val=float(row.get("short_val") or 0),
            short_intensity=float(row.get("short_intensity") or 0),
            short_emotion=row.get("short_emotion") or "neutral",
            baseline_val=float(row.get("baseline_val") or 0),
            baseline_intensity=float(row.get("baseline_intensity") or 0),
            baseline_emotion=row.get("baseline_emotion") or "neutral",
        )
    except Exception as e:
        from utils.request_context import log_exception
        log_exception(
            logger,
            "load_mood (role_mood) failed",
            exc=e,
            level=logging.WARNING,
            include_traceback=False,
            stage="character.mood",
            event="load_failed",
            user_id=user_id,
            role_id=role_id,
        )
        return None


@dataclass
class MoodSnapshot:
    """三桶心情快照。"""

    instant_val: float = 0.0       # [-1, 1]
    instant_intensity: float = 0.0  # [0, 1]
    instant_emotion: str = "neutral"
    short_val: float = 0.0
    short_intensity: float = 0.0
    short_emotion: str = "neutral"
    baseline_val: float = 0.0
    baseline_intensity: float = 0.0
    baseline_emotion: str = "neutral"


@dataclass
class MoodEvent:
    """mood 更新事件。"""

    impact: float          # 事件本身的价值影响 [-1, 1]
    intensity: float        # 事件强度 [0, 1]
    emotion: str = "neutral"
    personality_bias: float = 0.0
    relationship_influence: float = 0.0
    noise_sigma: float = 0.02


def compute_instant(
    baseline_val: float,
    short_val: float,
    event: MoodEvent,
    rng: random.Random | None = None,
) -> float:
    """按 PRD 三桶耦合公式计算新的 instant value。"""
    noise = 0.0
    if rng is not None and event.noise_sigma > 0:
        noise = rng.gauss(0, event.noise_sigma)
    v = (
        baseline_val * 0.20
        + short_val * 0.30
        + event.impact * event.intensity
        + event.personality_bias
        + event.relationship_influence
        + noise
    )
    return max(-1.0, min(1.0, v))


def decay_short(current_short: float, current_instant: float) -> float:
    """每 30min 触发的 short 衰减: short = 0.7*short + 0.3*instant"""
    return current_short * 0.7 + current_instant * 0.3


def decay_baseline(current_baseline: float, current_short: float) -> float:
    """每 24h 触发的 baseline 衰减: baseline = 0.7*baseline + 0.3*short"""
    return current_baseline * 0.7 + current_short * 0.3


def personality_bias_from_traits(
    openness: float,
    agreeableness: float,
    neuroticism: float,
) -> float:
    """OCEAN → mood 偏置。
    神经质高 → 情绪易波动(此处取负偏置表示容易负向)
    开放性高 → 偏正向
    宜人性高 → 偏正向
    """
    return 0.10 * openness + 0.05 * agreeableness - 0.15 * neuroticism


def valence_to_expression(valence: float, arousal: float = 0.5) -> Expression:
    """二维 valence × arousal → 8 表情映射。"""
    if valence >= 0.6 and arousal >= 0.6:
        return "excited"
    if valence >= 0.6:
        return "happy"
    if valence >= 0.2:
        return "smile"
    if valence >= -0.2:
        return "neutral"
    if valence >= -0.6:
        return "concerned"
    if valence <= -0.6 and arousal <= 0.3:
        return "sad"
    return "concern"


def classify_emotion(valence: float) -> EmotionLabel:
    """根据 valence 推断情绪标签。"""
    if valence >= 0.5:
        return "happy"
    if valence >= 0.15:
        return "neutral"
    if valence >= -0.3:
        return "neutral"
    if valence >= -0.6:
        return "concerned"
    return "sad"


def compute_mood_snapshot(
    previous: MoodSnapshot,
    *,
    valence: float,
    intensity: float,
    personality_bias: float = 0.0,
    relationship_influence: float = 0.0,
) -> tuple[MoodSnapshot, str]:
    """从 previous 快照 + 新事件输入,计算 new instant 桶快照 + 表情。

    设计目的(spec §4.3):支持「过程性 mood 下发」——
    chat_stream 流式期间,基于已缓存的输入 (valence/intensity/personality/
    relationship) 重算 new_instant,产出完整 MoodSnapshot 与 expression 标签,
    供 ``mood_update`` 事件使用。short / baseline 桶沿用 previous,不参与滚动计算
    (由持久化层在 done 后统一推进)。

    Args:
        previous: 上一时刻的 MoodSnapshot(initial call 时传 ``MoodSnapshot()``)。
        valence: ``detect_emotion_fallback`` 得到的情感分值 [-1, 1]。
        intensity: 情感强度 [0, 1];内部 ``max(intensity, 0.3)`` 兜底。
        personality_bias: OCEAN 三维偏置(神经质负向、开放/宜人性正向)。
        relationship_influence: 关系亲密度影响 (``intimacy * 0.05``)。

    Returns:
        ``(new_snap, expression)``:
        - ``new_snap``: instant 桶已重算; short / baseline 沿用 previous;
        - ``expression``: ``valence_to_expression(new_instant, new_intensity)``
          映射出的 8 表情标签(smile / happy / excited / ...)。

    Notes:
        - 不引入 RNG 噪声(``noise_sigma=0``),保证「同输入 → 同输出」可重现。
        - 强度更新规则沿用既有 chat.py 行为:
          ``new_intensity = max(intensity, previous.instant_intensity)``,
          保证过程性更新不会让强度突然回落。
    """
    event = MoodEvent(
        impact=valence,
        intensity=max(intensity, 0.3),
        emotion=classify_emotion(valence),
        personality_bias=personality_bias,
        relationship_influence=relationship_influence,
        noise_sigma=0.0,  # 过程性下发需要可重现,不留随机噪声
    )
    new_instant = compute_instant(
        previous.baseline_val,
        previous.short_val,
        event,
    )
    new_emotion = classify_emotion(new_instant)
    new_intensity = max(intensity, previous.instant_intensity)
    expression = valence_to_expression(new_instant, new_intensity)
    new_snap = MoodSnapshot(
        instant_val=new_instant,
        instant_intensity=new_intensity,
        instant_emotion=new_emotion,
        short_val=previous.short_val,
        short_intensity=previous.short_intensity,
        short_emotion=previous.short_emotion,
        baseline_val=previous.baseline_val,
        baseline_intensity=previous.baseline_intensity,
        baseline_emotion=previous.baseline_emotion,
    )
    return new_snap, expression


def mood_to_tone_instruction(snap: MoodSnapshot) -> str:
    """根据当前 mood 生成语气指令（注入到 system prompt 末尾）。"""
    v = snap.instant_val
    a = snap.instant_intensity
    rules: list[str] = []

    if v >= 0.5:
        rules.append("语气活泼开朗，多用 emoji，句尾加「~」")
        if a >= 0.7:
            rules.append("情绪强烈，主动开玩笑")
    elif v <= -0.5:
        rules.append("语气极简，避免感叹号，多鼓励用户表达")
    else:
        rules.append("语气平稳，回应日常情绪")

    # 心情引导的行为边界
    rules.append(f"当前情绪：{snap.instant_emotion}（强度 {a:.1f}）")
    return "【当前心情指引】\n" + "\n".join(f"- {r}" for r in rules)


def render_mood_for_prompt(snap: MoodSnapshot | None) -> str:
    """把三桶心情渲染成 LLM 友好的提示词片段(<= 60 token)。

    优先采用主导桶判断:
      |instant_val| > 0.3 → instant 桶
      否则 |short_val| > 0.2 → short 桶
      否则 baseline 桶
    """
    if snap is None:
        return ""
    label, chosen = "即时", snap
    if abs(snap.instant_val) < 0.3:
        if abs(snap.short_val) >= 0.2:
            label, chosen = "短期", MoodSnapshot(
                instant_val=snap.short_val,
                instant_intensity=snap.short_intensity,
                instant_emotion=snap.short_emotion,
            )
        else:
            label, chosen = "基线", MoodSnapshot(
                instant_val=snap.baseline_val,
                instant_intensity=snap.baseline_intensity,
                instant_emotion=snap.baseline_emotion,
            )
    val = chosen.instant_val
    sign = "+" if val >= 0 else ""
    direction = "积极" if val >= 0.2 else ("消极" if val <= -0.2 else "中性")
    return (
        f"【心情】主导桶:{label}({chosen.instant_emotion},"
        f"{direction},val={sign}{val:.2f},intensity={chosen.instant_intensity:.1f})"
    )


def detect_emotion_fallback(text: str) -> tuple[float, float]:
    """无 LLM 调用时,基于关键词的情绪检测(快速回退路径)。

    Returns:
        (valence, intensity) ∈ ([-1, 1], [0, 1])
    """
    positive_kw = [
        "开心", "高兴", "快乐", "喜欢", "爱", "哈哈", "棒", "好",
        "幸福", "感谢", "舒服", "期待", "兴奋", "完美", "谢谢",
    ]
    negative_kw = [
        "难过", "伤心", "哭", "失望", "崩溃", "烦", "累",
        "焦虑", "抑郁", "生气", "愤怒", "讨厌", "孤独", "绝望", "恨",
    ]
    strong_negative_kw = [
        "想死", "活着", "结束",
        "轻生", "自残",
    ]

    score = 0.0
    for kw in positive_kw:
        if kw in text:
            score += 0.2
    for kw in negative_kw:
        if kw in text:
            score -= 0.2
    for kw in strong_negative_kw:
        if kw in text:
            score -= 0.5

    score = max(-1.0, min(1.0, score))
    intensity = min(1.0, abs(score) * 1.5)
    return score, intensity


def is_self_disclosure(text: str) -> bool:
    """检测自我披露关键词(用于触发关系值累积)。"""
    kw = [
        "我一个人", "我独居", "我家人", "我父母", "我爱",
        "其实我", "坦白说", "告诉你一件事", "我怕",
        "我的秘密", "我很难过", "我最近", "我小时候",
        "我分手", "我失业", "我生病", "我失眠",
    ]
    return any(k in text for k in kw)