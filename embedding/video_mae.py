from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

import torch
from PIL import Image

from config.config import get_settings

if TYPE_CHECKING:
    from config.config import VideoMAESettings

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_clip_model: object | None = None
_clip_processor: object | None = None
_clip_device: str | None = None


def _resolve_device(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_clip_internal(device: str = "auto") -> tuple[object, object, str]:
    global _clip_model, _clip_processor, _clip_device
    target_device = _resolve_device(device)
    with _model_lock:
        if _clip_model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor

                clip_name = "openai/clip-vit-base-patch32"
                logger.info("Loading CLIP model=%s on device=%s for video embedding", clip_name, target_device)
                _clip_model = CLIPModel.from_pretrained(clip_name)
                _clip_processor = CLIPProcessor.from_pretrained(clip_name)
                _clip_model.eval()
                _clip_model.to(target_device)
                _clip_device = target_device
                logger.info("CLIP model loaded for video embedding")
            except Exception as e:
                logger.error("Failed to load CLIP model: %s", e)
                raise
    return _clip_model, _clip_processor, _clip_device


def _extract_frames_opencv(video_path: str, sample_rate: int, max_frames: int) -> list[Image.Image]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps > 0:
            duration = total_frames / fps
            sample_rate = max(1, int(total_frames / min(max_frames, int(duration / 2))))
        else:
            sample_rate = max(1, sample_rate)
        frames: list[Image.Image] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                if len(frames) >= max_frames:
                    break
            frame_idx += 1
        return frames
    finally:
        cap.release()


def _extract_frames_imageio(video_path: str, sample_rate: int, max_frames: int) -> list[Image.Image]:
    import imageio

    reader = imageio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        total_frames = meta.get("nframes", 0)
        fps = meta.get("fps", 0)
        if fps > 0 and total_frames > 0:
            duration = total_frames / fps
            sample_rate = max(1, int(total_frames / min(max_frames, int(duration / 2))))
        else:
            sample_rate = max(1, sample_rate)
        frames: list[Image.Image] = []
        for idx, frame in enumerate(reader):
            if idx % sample_rate == 0:
                frames.append(Image.fromarray(frame))
                if len(frames) >= max_frames:
                    break
        return frames
    finally:
        reader.close()


def _extract_frames(video_path: str, sample_rate: int, max_frames: int) -> list[Image.Image]:
    try:
        return _extract_frames_opencv(video_path, sample_rate, max_frames)
    except Exception:
        logger.info("opencv frame extraction failed, falling back to imageio")
        try:
            return _extract_frames_imageio(video_path, sample_rate, max_frames)
        except Exception as e:
            raise RuntimeError(f"Failed to extract frames from video: {e}")


def _encode_frames(frames: list[Image.Image], device: str = "auto") -> list[list[float]]:
    model, processor, _ = _load_clip_internal(device)
    try:
        with torch.no_grad():
            inputs = processor(images=frames, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return image_features.cpu().tolist()
    except Exception as e:
        logger.error("_encode_frames failed: %s", e)
        raise


def compute_video_embedding(video_path: str, device: str = "auto") -> list[list[float]]:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    settings: VideoMAESettings = get_settings().multimodal.video_mae
    frames = _extract_frames(video_path, settings.frame_sample_rate, settings.max_frames)
    if not frames:
        logger.warning("No frames extracted from video: %s", video_path)
        return []
    return _encode_frames(frames, device=device or settings.device)


def compute_video_summary_embedding(video_path: str, device: str = "auto") -> list[float]:
    frame_embeddings = compute_video_embedding(video_path, device=device)
    if not frame_embeddings:
        return []
    dim = len(frame_embeddings[0])
    avg = [0.0] * dim
    for vec in frame_embeddings:
        for i in range(dim):
            avg[i] += vec[i]
    n = len(frame_embeddings)
    avg = [v / n for v in avg]
    norm = sum(v * v for v in avg) ** 0.5
    if norm > 0:
        avg = [v / norm for v in avg]
    return avg