"""视频解析器：抽关键帧 → 视觉模型逐帧描述 → 汇总。

修正原 video_mae 实现的 cv2.imdecode 错误（实际是图片解码，不是视频解码）；
改用 cv2.VideoCapture 抽取帧。
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile

import cv2  # type: ignore

from llm.client import get_llm_client
from utils.downloader import fetch_source_bytes
from utils.request_context import log_exception, merge_extra
from utils.temp_files import safe_remove
from .base import ParsedFile, ParsedChunk

logger = logging.getLogger(__name__)

_KEYFRAME_PROMPT = (
    "这是从一段视频中抽出的一个关键帧。请用 1~2 段中文描述这一帧：\n"
    "- 画面主体（人物/场景/动作）\n"
    "- 显著视觉细节（文字、表情、地标）\n"
    "如果多帧串起来，请关注场景/人物/事件的变化。只描述肉眼可见的内容。"
)

MAX_KEYFRAMES = 8


async def _describe_frame(image_path: str) -> str:
    """单帧视觉描述：读取为 base64 data url 后走视觉模型。

    直接走 file:// 不行（LLM 不访问本地），所以用 data URL 嵌入。
    """
    import base64

    with open(image_path, "rb") as f:
        b = base64.b64encode(f.read()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b}"
    client = get_llm_client()
    messages = [
        {"role": "system", "content": _KEYFRAME_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这一关键帧。"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    try:
        resp = await client.chat(messages, max_tokens=400, temperature=0.4)
        try:
            return (resp["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return ""
    except Exception as e:
        log_exception(
            logger,
            "frame vision failed",
            exc=e,
            level=logging.WARNING,
            stage="parser_video",
            event="vision_error",
        )
        return ""


def _extract_keyframes(video_path: str, out_dir: str, max_n: int = MAX_KEYFRAMES) -> list[str]:
    """用 cv2.VideoCapture 均匀抽 max_n 帧，保存为 jpg，返回路径列表。"""
    paths: list[str] = []
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            # 退化：取首帧
            ret, frame = cap.read()
            if ret:
                p = os.path.join(out_dir, "kf_0.jpg")
                cv2.imwrite(p, frame)
                paths.append(p)
            return paths
        n = min(max_n, total)
        step = max(1, total // n)
        idx = 0
        saved = 0
        while saved < n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            p = os.path.join(out_dir, f"kf_{saved}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p)
            saved += 1
            idx += step
    finally:
        cap.release()
    return paths


async def parse_video(file_key: str, file_name: str, url: str | None) -> ParsedFile:
    if not file_key and not url:
        return ParsedFile(modality="video", meta={"error": "no_key_or_url"})
    workdir: str | None = None
    try:
        # fileKey 优先直连七牛，url 仅兜底
        raw = await fetch_source_bytes(file_key or None, url or None)
        workdir = tempfile.mkdtemp(prefix="echo-video-")
        video_path = os.path.join(workdir, "src.mp4")
        with open(video_path, "wb") as f:
            f.write(raw)
        # 抽帧（同步，丢线程池）
        frames = await asyncio.to_thread(_extract_keyframes, video_path, workdir)
        if not frames:
            return ParsedFile(modality="video", meta={"error": "no_frames"})
        # 逐帧视觉描述
        descriptions: list[str] = []
        for i, fp in enumerate(frames):
            d = await _describe_frame(fp)
            if d:
                descriptions.append(f"[关键帧 {i + 1}/{len(frames)}] {d}")
        if not descriptions:
            return ParsedFile(modality="video", meta={"error": "vision_failed"})
        full = "\n\n".join(descriptions)
        logger.info(
            "video parsed",
            extra=merge_extra(
                stage="parser_video",
                event="ok",
                file_name=file_name,
                frames=len(frames),
                desc_len=len(full),
            ),
        )
        return ParsedFile(
            modality="video",
            text=full[:3000],
            chunks=[ParsedChunk(text=full, source=file_name)],
            detail_md=full,
            meta={"frames": len(frames), "bytes": len(raw)},
        )
    except Exception as e:
        log_exception(
            logger,
            "parse_video failed",
            exc=e,
            level=logging.WARNING,
            stage="parser_video",
            event="error",
            file_name=file_name,
        )
        return ParsedFile(modality="video", meta={"error": str(e)[:200]})
    finally:
        if workdir:
            safe_remove(workdir)