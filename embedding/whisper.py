"""Whisper 音频 embedding：转录 + 声纹嵌入。

- 转录：`openai-whisper`（如未安装则返回空串与提示）。档位/语种/繁简由
  `WhisperSettings`（`WHISPER_` 前缀）控制，默认 medium + zh + 繁转简。
- 声纹：使用 BGE-M3 对转录文本编码作为弱声纹特征；这样不依赖额外的声纹模型，
  且便于跨模态检索对齐。也可替换为专门的 speaker embedding（如 resemblyzer）。

设计原则：
- 模型加载失败时**绝不抛异常**，返回 `(text="", embedding=zero_vec)`，
  让上层调用方可以正常继续。
- 不依赖 "ffmpeg" 这个名字能被 PATH 解析：自己拿绝对路径解码成 16kHz 单声道
  float32 再喂给 whisper，绕开其内部按名字找 ffmpeg 的 subprocess 调用。
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from config.config import get_settings
from utils.model_cache import ModelNotCachedError, apply_hf_env, resolve_whisper_root
from utils.request_context import log_exception, log_silent_failure, merge_extra

logger = logging.getLogger(__name__)

# whisper 的编码器固定吃 16kHz 单声道；解码参数必须与之对齐。
_WHISPER_SAMPLE_RATE = 16000


def _apply_endpoint_env() -> None:
    """固化 HF 镜像 / 超时 / 缓存根环境变量；必须在首次 HF 请求前调用。"""
    cfg = get_settings().embedding
    apply_hf_env(endpoint=cfg.endpoint, download_timeout=cfg.download_timeout)


def _resolve_ffmpeg() -> str | None:
    """解析出 ffmpeg 可执行文件的**绝对路径**，拿不到返回 None。

    为什么不走"把目录塞进 PATH"那条路（这正是之前的 bug）：
      imageio-ffmpeg 装出来的文件叫 `ffmpeg-win-x86_64-v7.1.exe`，**不叫**
      `ffmpeg.exe`。把它所在目录加进 PATH 毫无意义——目录里压根没有叫 `ffmpeg`
      的东西。而 openai-whisper 的 `audio.load_audio` 硬编码 `cmd[0] = "ffmpeg"`，
      靠 PATH 按名字查找，于是必然 `FileNotFoundError [WinError 2]`。
      `FFMPEG_BINARY` 同样无效：那是 moviepy 的约定，whisper 从不读它。

    所以这里只回传绝对路径，由 `_decode_audio` 直接调用，不依赖任何名字解析。
    优先用用户自己装的 ffmpeg，其次才用 imageio-ffmpeg 内置版本。
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.exists(bundled):
            logger.info(
                "ffmpeg resolved via imageio_ffmpeg",
                extra=merge_extra(
                    stage="whisper_transcribe",
                    event="ffmpeg_bundled",
                    ffmpeg_path=bundled,
                ),
            )
            return bundled
    except Exception as e:
        log_silent_failure(
            logger,
            "imageio_ffmpeg not available; audio transcription will be skipped",
            exc=e,
            stage="whisper_transcribe",
            event="ffmpeg_missing",
        )
    return None


def _decode_audio(path: str, exe: str):
    """用绝对路径的 ffmpeg 把任意音频解码成 whisper 需要的波形。

    输出与 `whisper.audio.load_audio` 完全一致：16kHz 单声道 float32，
    归一化到 [-1, 1]。这样可以把数组直接喂给 `model.transcribe()`
    （它接受 `str | np.ndarray | torch.Tensor`），绕开其内部那次按名字找
    ffmpeg 的 subprocess 调用。
    """
    import numpy as np

    cmd = [
        exe,
        "-nostdin",
        "-threads", "0",
        "-i", path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(_WHISPER_SAMPLE_RATE),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        # 必须带上 stderr：否则线上只能看到一句没有信息量的"解码失败"
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed (rc={proc.returncode}): {stderr[-500:]}")
    return np.frombuffer(proc.stdout, np.int16).flatten().astype(np.float32) / 32768.0


# ---------- 繁 → 简 ----------

# opencc 转换器缓存。None = 尚未尝试加载；False = 加载失败（不再重试）。
_OPENCC: object | None | bool = None


def _reset_opencc_cache() -> None:
    """测试用：清掉 opencc 缓存，让下次 _to_simplified 重新尝试加载。"""
    global _OPENCC
    _OPENCC = None


def _to_simplified(text: str) -> str:
    """繁体 → 简体。opencc 不可用时原样返回（转录本身仍有价值，不该因此失败）。

    为什么必须做这一步：whisper 中文倾向输出繁体，small 以上尤其明显
    （实测 small 输出"他昨天吃壞了肚子,上吐下瀉...這隻小貓真的很可愛"）。
    为什么不用 initial_prompt 引导简体：那是在解码阶段施加偏好，会干扰选字——
    实测会把"上吐下**泻**"带偏成"上吐下**泄**"。opencc 是确定性的字表映射，
    只改字形不改选字，实测能把"上吐下瀉/這隻"精确还原为"上吐下泻/这只"。
    """
    global _OPENCC
    if not text:
        return text
    if _OPENCC is False:
        return text
    if _OPENCC is None:
        try:
            from opencc import OpenCC  # type: ignore

            _OPENCC = OpenCC("t2s")
        except Exception as e:
            _OPENCC = False
            log_silent_failure(
                logger,
                "opencc unavailable; keep original (traditional) transcript",
                exc=e,
                stage="whisper_transcribe",
                event="opencc_missing",
            )
            return text
    try:
        return _OPENCC.convert(text)  # type: ignore[union-attr]
    except Exception as e:
        log_silent_failure(
            logger,
            "opencc convert failed; keep original transcript",
            exc=e,
            stage="whisper_transcribe",
            event="opencc_error",
        )
        return text


# ---------- 模型 ----------

# 按档位缓存已加载的模型：medium 加载要数秒、占 1.5G 内存，绝不能每次转录都重加载。
_MODELS: dict[str, object] = {}


def _reset_model_cache() -> None:
    """测试用：清掉模型缓存。"""
    _MODELS.clear()


def _get_model(name: str):
    """加载并缓存指定档位的 whisper 模型。

    权重优先从指定缓存目录（MODEL_CACHE_DIR/whisper）获取；未命中且允许下载时，
    交由 whisper 内部按 download_root 下载。禁止下载（MODEL_AUTO_DOWNLOAD=false）
    且本地缺失时抛 ModelNotCachedError，由上层走空转录兜底。
    """
    cached = _MODELS.get(name)
    if cached is not None:
        return cached
    import whisper  # type: ignore

    download_root = resolve_whisper_root()
    if not get_settings().model.auto_download and download_root:
        pt = Path(download_root) / f"{name}.pt"
        if not pt.exists():
            raise ModelNotCachedError(
                f"whisper model {name!r} not in cache dir and auto_download disabled "
                f"(cache_dir={download_root})"
            )
    t0 = time.perf_counter()
    model = whisper.load_model(name, download_root=download_root)
    _MODELS[name] = model
    logger.info(
        "whisper model loaded",
        extra=merge_extra(
            stage="whisper_transcribe",
            event="model_loaded",
            model=name,
            download_root=download_root or "default",
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        ),
    )
    return model


def _fallback_embedding(text: str, dim: int) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals: list[float] = []
    i = 0
    while len(vals) < dim:
        vals.append((h[i % len(h)] / 255.0) * 2.0 - 1.0)
        i += 1
    return vals[:dim]


def _transcribe(audio_bytes: bytes) -> str:
    try:
        cfg = get_settings().whisper

        # 必须在 whisper.load_model 之前把 HF 镜像设置好，否则它内部
        # 下载权重时会直连 huggingface.co → SSL 失败。
        _apply_endpoint_env()

        exe = _resolve_ffmpeg()
        if not exe:
            logger.warning(
                "ffmpeg unavailable; skip transcription",
                extra=merge_extra(
                    stage="whisper_transcribe",
                    event="ffmpeg_unavailable",
                    audio_bytes=len(audio_bytes or b""),
                ),
            )
            return ""

        model = _get_model(cfg.model)

        # ffmpeg 需要可 seek 的输入（m4a/mp4 的 moov 可能在文件尾），所以先落盘。
        # 后缀不影响解码——ffmpeg 按内容嗅探格式。
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            audio = _decode_audio(tmp_path, exe)
            t0 = time.perf_counter()
            # language 留空表示自动检测；显式指定可省掉一次检测且避免误判语种。
            result = model.transcribe(audio, fp16=False, language=cfg.language or None)
            raw = (result.get("text") or "").strip()
            text = _to_simplified(raw) if cfg.simplified else raw
            logger.info(
                "whisper transcribe ok",
                extra=merge_extra(
                    stage="whisper_transcribe",
                    event="ok",
                    model=cfg.model,
                    language=result.get("language") or cfg.language,
                    audio_secs=round(len(audio) / _WHISPER_SAMPLE_RATE, 2),
                    text_len=len(text),
                    converted=bool(cfg.simplified and text != raw),
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                ),
            )
            return text
        finally:
            try:
                os.remove(tmp_path)
            except Exception as e:
                log_silent_failure(
                    logger,
                    "whisper temp file cleanup failed (skip)",
                    exc=e,
                    stage="whisper_transcribe",
                    event="tmp_cleanup_error",
                    tmp_path=tmp_path,
                )
    except Exception as e:
        log_exception(
            logger,
            "whisper transcribe failed (return empty text)",
            exc=e,
            level=logging.WARNING,
            stage="whisper_transcribe",
            event="transcribe_error",
            audio_bytes=len(audio_bytes or b""),
        )
        return ""


def embed_audio(audio_bytes: bytes) -> dict:
    """音频 → {text: str, embedding: list[float]}。失败返回空文本+零向量。"""
    cfg = get_settings()
    dim = cfg.bge_m3.dim
    t0 = time.perf_counter()
    text = _transcribe(audio_bytes) if audio_bytes else ""
    if text:
        from embedding import bge_m3

        embedding = bge_m3.embed_texts([text])[0]
    else:
        embedding = _fallback_embedding("", dim)
    logger.info(
        "whisper.embed_audio done",
        extra=merge_extra(
            stage="whisper_embed",
            event="ok",
            audio_bytes=len(audio_bytes) if audio_bytes else 0,
            text_len=len(text),
            dim=len(embedding),
            duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        ),
    )
    return {"text": text, "embedding": embedding, "dim": dim}


def dim() -> int:
    return get_settings().bge_m3.dim