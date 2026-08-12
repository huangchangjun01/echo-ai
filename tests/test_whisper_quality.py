"""回归测试：中文转录质量相关的配置与后处理。

背景（用户反馈）：
  ffmpeg 修好、音频能转录之后，识别文本错别字严重——
  "上吐下泻" → "上图下险"，"可爱" → "可啊"。

实测复现（10.4s 中文 TTS 音频，原文"她昨天吃坏了肚子，上吐下泻，难受了一整晚。
这只小猫真的很可爱，我特别喜欢它。"）：

  base（旧默认）  1.7s  他昨天吃坏了肚子,上**土**下**泄**,难受了一整**碗**。这**支**小猫...
  small           2.7s  他昨天吃**壞**了肚子,上吐下**瀉**,難受了一整晚。這**隻**小貓...
  medium          9.2s  他昨天吃坏了肚子,上吐下泻,难受了一整晚。这只小猫真的很可爱...

两个独立根因：
  1. `whisper.load_model("base")` 写死。base 是 74M 多语种模型，中文最弱，
     同音字选错就是它的典型failure mode。且全项目没有任何 whisper 配置项。
  2. 换成 small 以上后 whisper 会大量输出**繁体**。若只改模型不做繁简处理，
     只是把"错别字"换成"满屏繁体"。
     用 `initial_prompt` 引导简体不可靠：实测会把"上吐下**泻**"带偏成"上吐下**泄**"
     （解码阶段施加偏好会干扰选字）。故采用 opencc 做识别后的确定性转换。

本测试覆盖配置透传与繁简转换；真实识别质量依赖模型下载且很慢，不进 CI。
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import embedding.whisper as whisper_mod
from config.config import get_settings


@pytest.fixture
def fake_whisper(monkeypatch):
    """把 whisper 换成可观测的替身，记录 load_model / transcribe 收到的参数。"""
    seen: dict = {}

    class _FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def transcribe(self, audio, **kwargs):
            seen["audio"] = audio
            seen["kwargs"] = kwargs
            return {"text": seen.get("_text", "")}

    def _load_model(name, **kwargs):
        seen["model"] = name
        seen["load_kwargs"] = kwargs
        return _FakeModel(name)

    fake = type(sys)("whisper")
    fake.load_model = _load_model
    monkeypatch.setitem(sys.modules, "whisper", fake)
    monkeypatch.setattr(whisper_mod, "_resolve_ffmpeg", lambda: "ffmpeg-stub")
    monkeypatch.setattr(
        whisper_mod, "_decode_audio", lambda path, exe: np.zeros(16000, dtype=np.float32)
    )
    # 模型按档位缓存；每个用例必须从干净状态开始，否则会拿到上一个用例的替身
    whisper_mod._reset_model_cache()
    yield seen
    whisper_mod._reset_model_cache()


# ---------- 配置 ----------

def test_default_model_is_medium_not_base():
    """默认档位必须是 medium：base 的中文错别字正是本 bug 的主因。"""
    cfg = get_settings().whisper
    assert cfg.model == "medium"


def test_default_language_and_simplified():
    """默认显式指定中文并开启繁转简。"""
    cfg = get_settings().whisper
    assert cfg.language == "zh"
    assert cfg.simplified is True


def test_transcribe_uses_configured_model(fake_whisper, monkeypatch):
    monkeypatch.setattr(get_settings().whisper, "model", "small")

    whisper_mod._transcribe(b"audio")

    assert fake_whisper["model"] == "small"


def test_transcribe_passes_language_explicitly(fake_whisper):
    """必须显式传 language：省掉一次语种自动检测，也避免短音频误判语种输出乱码。"""
    whisper_mod._transcribe(b"audio")

    assert fake_whisper["kwargs"].get("language") == "zh"


def test_empty_language_means_autodetect(fake_whisper, monkeypatch):
    """配置留空时退回自动检测，不要硬塞一个空字符串给 whisper。"""
    monkeypatch.setattr(get_settings().whisper, "language", "")

    whisper_mod._transcribe(b"audio")

    assert fake_whisper["kwargs"].get("language") is None


# ---------- 繁转简 ----------

def test_traditional_output_is_converted_to_simplified(fake_whisper):
    """核心回归：whisper 的繁体输出必须落地成简体。"""
    fake_whisper["_text"] = "他昨天吃壞了肚子,上吐下瀉,難受了一整晚。這隻小貓真的很可愛"

    text = whisper_mod._transcribe(b"audio")

    assert "上吐下泻" in text
    assert "这只小猫" in text
    assert "可爱" in text
    for bad in ("壞", "瀉", "這隻", "小貓", "可愛"):
        assert bad not in text, f"仍残留繁体字: {bad}"


def test_simplified_disabled_keeps_original(fake_whisper, monkeypatch):
    monkeypatch.setattr(get_settings().whisper, "simplified", False)
    fake_whisper["_text"] = "這隻小貓真的很可愛"

    assert whisper_mod._transcribe(b"audio") == "這隻小貓真的很可愛"


def test_to_simplified_degrades_when_opencc_missing(monkeypatch):
    """opencc 缺失时原样返回，绝不抛异常——转录本身仍然有价值。"""
    monkeypatch.setitem(sys.modules, "opencc", None)
    whisper_mod._reset_opencc_cache()

    assert whisper_mod._to_simplified("這隻小貓") == "這隻小貓"

    whisper_mod._reset_opencc_cache()


def test_to_simplified_is_noop_for_already_simplified():
    assert whisper_mod._to_simplified("这只小猫真的很可爱") == "这只小猫真的很可爱"


def test_to_simplified_handles_empty():
    assert whisper_mod._to_simplified("") == ""
