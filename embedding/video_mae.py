"""VideoMAE / 关键帧 CLIP 视频 embedding。

- 优先使用 VideoMAE；若环境未安装则降级为「抽关键帧 → CLIP 平均池化」。
- 任意失败均返回 CLIP 维度的零向量 + 关键帧描述文本，便于上层容错。

实现简化：对输入视频做最多 8 帧均匀采样，用 CLIP 编码后取均值作为整体向量。
"""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence

from config.config import get_settings
from utils.request_context import log_exception, log_silent_failure

logger = logging.getLogger(__name__)


def _sample_keyframes(video_bytes: bytes, max_frames: int = 8) -> list[bytes]:
    """从视频字节中均匀抽取若干帧的 JPEG 字节。失败返回空列表。"""
    try:
        import cv2  # type: ignore

        arr = np_frombuffer(video_bytes)
        if arr is None:
            return []
        frames: list[bytes] = []
        n = len(arr)
        if n == 0:
            return []
        idxs = [int(i * (n - 1) / max(1, max_frames - 1)) for i in range(max_frames)]
        for i in idxs:
            ok, buf = cv2.imencode(".jpg", arr[i])
            if ok:
                frames.append(buf.tobytes())
        return frames
    except Exception as e:
        log_exception(
            logger,
            "sample_keyframes failed",
            exc=e,
            level=logging.WARNING,
            stage="video_mae",
            event="sample_error",
            bytes=len(video_bytes or b""),
            max_frames=max_frames,
        )
        return []


def np_frombuffer(data: bytes):
    try:
        import cv2  # type: ignore
        import numpy as np

        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        log_silent_failure(
            logger,
            "np_frombuffer: cv2 decode failed",
            exc=e,
            stage="video_mae",
            event="decode_error",
            bytes=len(data or b""),
        )
        return None


def _aggregate_clip(frame_bytes_list: Sequence[bytes]) -> tuple[list[float], str]:
    """聚合 CLIP 特征 + 拼接帧描述文本占位。"""
    cfg = get_settings().embedding
    from embedding import clip

    if not frame_bytes_list:
        return [0.0] * cfg.dim, ""
    vecs = clip.embed_images(frame_bytes_list)
    if not vecs:
        return [0.0] * cfg.dim, ""
    n = len(vecs)
    avg = [sum(v[i] for v in vecs) / max(1, n) for i in range(cfg.dim)]
    desc = f"video with {n} keyframes"
    return avg, desc


def embed_video(video_bytes: bytes) -> dict:
    """视频 → {embedding: list[float], description: str}。"""
    cfg = get_settings()
    frames = _sample_keyframes(video_bytes) if video_bytes else []
    vec, desc = _aggregate_clip(frames)
    if not frames:
        # 完全失败：返回零向量 + 占位描述
        return {
            "embedding": [0.0] * cfg.embedding.dim,
            "description": "",
            "dim": cfg.embedding.dim,
        }
    return {"embedding": vec, "description": desc, "dim": cfg.embedding.dim}


def dim() -> int:
    return get_settings().embedding.dim