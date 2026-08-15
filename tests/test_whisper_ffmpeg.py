"""回归测试：whisper 转录必须能真正调起 ffmpeg。

背景（线上故障）：
  音频下载成功（1447645 bytes）后，`model.transcribe(tmp_path)` 抛
  `FileNotFoundError: [WinError 2] 系统找不到指定的文件。`
  栈底是 `whisper/audio.py::load_audio` → `subprocess.run(["ffmpeg", ...])`。

根因：
  imageio-ffmpeg 装出来的可执行文件叫 **ffmpeg-win-x86_64-v7.1.exe**，不叫
  `ffmpeg.exe`。老实现 `_ensure_ffmpeg_available` 只是把它所在的**目录**塞进
  PATH，可那个目录里根本没有 `ffmpeg.exe`；而 whisper 的 `load_audio` 硬编码
  `cmd[0] = "ffmpeg"`，靠 PATH 按名字找 → 永远找不到。
  `FFMPEG_BINARY` 也没用：那是 moviepy 的约定，openai-whisper 从不读它。

  老测试之所以一直是绿的，是因为它只断言"PATH / FFMPEG_BINARY 被设置了"这个
  **机制**，从没断言"ffmpeg 真的能被拉起来"这个**结果**。下面第一个用例专门堵这个洞。

修复方向：
  不再依赖"ffmpeg"这个名字能被 PATH 解析。`_resolve_ffmpeg()` 拿到绝对路径，
  `_decode_audio()` 用绝对路径自己解码成 16kHz 单声道 float32，再把数组交给
  whisper（`transcribe` 接受 np.ndarray），彻底绕开 `load_audio`。
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import embedding.whisper as whisper_mod


def _has_imageio_ffmpeg() -> bool:
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        return False
    return True


def test_resolved_ffmpeg_is_actually_launchable():
    """核心回归：解析出来的 ffmpeg 必须能真的被 subprocess 拉起来。

    老实现在这里必挂：它交出的是一个"目录里并不存在 ffmpeg.exe"的 PATH。
    """
    if not _has_imageio_ffmpeg():
        pytest.skip("imageio-ffmpeg not installed in this env")

    exe = whisper_mod._resolve_ffmpeg()

    assert exe, "未能解析出 ffmpeg 可执行文件"
    proc = subprocess.run([exe, "-version"], capture_output=True)
    assert proc.returncode == 0, f"ffmpeg 无法执行: {proc.stderr[:200]!r}"
    assert b"ffmpeg version" in proc.stdout.lower()


def test_resolve_ffmpeg_prefers_real_ffmpeg_on_path(monkeypatch, tmp_path):
    """PATH 里已有真 ffmpeg 时优先用它，不要越俎代庖换成内置版本。"""
    name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    fake = tmp_path / name
    fake.write_text("")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))

    # Windows 上 shutil.which 会按 PATHEXT 归一化扩展名大小写，故忽略大小写比较
    assert whisper_mod._resolve_ffmpeg().lower() == str(fake).lower()

def test_resolve_ffmpeg_returns_none_when_nothing_available(monkeypatch):
    """两条路都断了要静默返回 None，让上层走"转录失败"兜底，而不是抛异常。"""
    monkeypatch.setenv("PATH", "")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)

    assert whisper_mod._resolve_ffmpeg() is None


def test_decode_audio_returns_normalized_mono_float32(tmp_path):
    """解码结果必须是 whisper 要的格式：16kHz 单声道 float32，取值在 [-1, 1]。"""
    if not _has_imageio_ffmpeg():
        pytest.skip("imageio-ffmpeg not installed in this env")

    import numpy as np

    exe = whisper_mod._resolve_ffmpeg()
    assert exe

    # 用 ffmpeg 自己造一段 0.5 秒正弦波做输入（避免往仓库塞二进制 fixture）
    src = tmp_path / "tone.wav"
    subprocess.run(
        [exe, "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5", "-y", str(src)],
        capture_output=True,
        check=True,
    )

    audio = whisper_mod._decode_audio(str(src), exe)

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1, "必须是单声道一维数组"
    # 16kHz * 0.5s ≈ 8000 采样点，容忍编码器少量补边
    assert 7000 < audio.size < 9000, f"采样率疑似不是 16kHz，实际长度 {audio.size}"
    assert float(np.abs(audio).max()) <= 1.0


def test_decode_audio_raises_with_ffmpeg_stderr_on_bad_input(tmp_path):
    """解码失败必须带上 ffmpeg 的 stderr，否则线上只能看到一句没有信息量的报错。"""
    if not _has_imageio_ffmpeg():
        pytest.skip("imageio-ffmpeg not installed in this env")

    exe = whisper_mod._resolve_ffmpeg()
    assert exe
    junk = tmp_path / "not-audio.bin"
    junk.write_bytes(b"definitely not audio" * 10)

    with pytest.raises(RuntimeError) as ei:
        whisper_mod._decode_audio(str(junk), exe)

    assert "ffmpeg" in str(ei.value).lower()


def test_transcribe_returns_empty_when_ffmpeg_missing(monkeypatch):
    """ffmpeg 彻底不可用时，_transcribe 返回空串而不是抛异常（上层靠空串判失败）。"""
    monkeypatch.setattr(whisper_mod, "_resolve_ffmpeg", lambda: None)

    assert whisper_mod._transcribe(b"fake audio bytes") == ""


def test_transcribe_feeds_decoded_array_not_path(monkeypatch):
    """回归核心：必须把解码好的数组喂给 whisper，绝不能再把路径交给它去调 ffmpeg。"""
    import numpy as np

    seen: dict = {}

    class _FakeModel:
        def transcribe(self, audio, **kwargs):
            seen["audio"] = audio
            return {"text": "  你好世界  "}

    fake_whisper = type(sys)("whisper")
    # 真实 whisper.load_model 支持 download_root（模型缓存目录），mock 需兼容该 kwarg
    fake_whisper.load_model = lambda name, **kwargs: _FakeModel()
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    monkeypatch.setattr(whisper_mod, "_resolve_ffmpeg", lambda: "ffmpeg-stub")
    monkeypatch.setattr(
        whisper_mod, "_decode_audio", lambda path, exe: np.zeros(16000, dtype=np.float32)
    )

    text = whisper_mod._transcribe(b"fake audio bytes")

    assert text == "你好世界"
    assert isinstance(seen["audio"], np.ndarray), (
        "whisper 收到的必须是解码后的数组；传路径会让它自己去 PATH 找 ffmpeg，正是本 bug 的成因"
    )
