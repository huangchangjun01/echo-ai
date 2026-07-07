"""Embedding 子包统一入口。

按模态暴露 4 个独立模块：
- `bge_m3`  文本 768 维
- `clip`    图像 512 维（CLIP）
- `whisper` 音频（转录 + BGE 嵌入）
- `video_mae` 视频（关键帧 CLIP 聚合）

同时保留旧接口以保证业务层平滑迁移：`ChineseCLIPEmbeddings`、`models`。
"""

from . import bge_m3, clip, video_mae, whisper
from .embeddings import ChineseCLIPEmbeddings
from .models import (
    compute_embedding,
    compute_image_embeddings,
    compute_text_embedding,
    compute_text_embeddings,
    detect_device,
    load_model,
    warmup,
)

__all__ = [
    "bge_m3",
    "clip",
    "video_mae",
    "whisper",
    "ChineseCLIPEmbeddings",
    "compute_embedding",
    "compute_image_embeddings",
    "compute_text_embedding",
    "compute_text_embeddings",
    "detect_device",
    "load_model",
    "warmup",
]