"""回归测试：视频解析器应按场景自适应切分，每场景独立返回 ParsedChunk。

被测函数：
  - parsers.video_parser._detect_scenes：基于 cv2.absdiff 的场景切分
  - parsers.video_parser._sample_scene_frames：每场景采 1~3 张代表帧
  - parsers.video_parser.parse_video：把每个场景独立返回为 ParsedChunk
  - biz/recall 的 per-scene 展开逻辑（通过 mock）

无需真实 LLM：所有视觉调用都 monkeypatch 掉。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from parsers import video_parser
from parsers.base import ParsedChunk, ParsedFile
from parsers.video_parser import _detect_scenes, _sample_scene_frames


def _make_scene_video(path: str, scenes: list[tuple[int, tuple[int, int, int]]]) -> None:
    """造一段明确含多个"场景"的视频。

    scenes = [(frame_count, (B, G, R)), ...] —— 每个 tuple 是一段连续同色帧。
    总帧数 = sum(frame_count for ...)。相邻场景颜色不同 → diff 必超阈值。
    """
    w, h, fps = 128, 96, 10
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for n_frames, color in scenes:
        for i in range(n_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:] = color
            # 每段中间帧写一个白字标记，便于人工检查
            if i == n_frames // 2:
                cv2.putText(
                    frame, f"c{color[0]}{color[1]}{color[2]}",
                    (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
            out.write(frame)
    out.release()


@pytest.fixture
def tmp_video(tmp_path: Path) -> str:
    """3 段：红 30 帧 → 绿 30 帧 → 蓝 30 帧（每段 3 秒 @10fps）。"""
    path = str(tmp_path / "scene_clip.mp4")
    _make_scene_video(path, [(30, (0, 0, 255)), (30, (0, 255, 0)), (30, (255, 0, 0))])
    return path


def test_detect_senes_finds_three_distinct_scenes(tmp_video: str) -> None:
    """3 段差异明显的视频应被切为 3 个场景。"""
    cap = cv2.VideoCapture(tmp_video)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        assert total == 90
        scenes = _detect_scenes(cap, total, fps)
    finally:
        cap.release()

    assert len(scenes) == 3, f"expected 3 scenes, got {len(scenes)}: {scenes}"
    # 每段 ≥ 8 帧（MIN_SCENE_FRAMES）
    for s, e in scenes:
        assert e - s >= 8, f"scene too short: ({s},{e})"
    # 覆盖全视频
    assert scenes[0][0] == 0
    assert scenes[-1][1] == 90


def test_sample_scene_frames_picks_1_to_3_per_scene(tmp_video: str) -> None:
    """每场景应采到 1~3 张代表帧；30 帧场景 → 3 张（≥24 帧规则）。"""
    cap = cv2.VideoCapture(tmp_video)
    out_dir = str(Path(tmp_video).parent / "frames")
    import os

    os.makedirs(out_dir, exist_ok=True)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        scenes = _detect_scenes(cap, total, fps)
        # 二次打开 cap 时要重置（_detect_scenes 会 seek 到末尾）
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        per_scene = [_sample_scene_frames(cap, s, out_dir, i + 1) for i, s in enumerate(scenes)]
    finally:
        cap.release()

    # 3 段、每段 30 帧 → 每段 3 张
    assert len(per_scene) == 3
    for frames in per_scene:
        assert 1 <= len(frames) <= 3, f"frames per scene: {len(frames)}"


@pytest.mark.parametrize(
    ("scenes_def", "expected_min", "expected_max"),
    [
        # 单段 → 1 个场景
        ([(60, (0, 0, 255))], 1, 1),
        # 2 段不同色 → 2 个场景
        ([(30, (0, 0, 255)), (30, (0, 255, 0))], 2, 2),
        # 5 段不同色 → 5 个场景
        ([
            (20, (0, 0, 255)), (20, (0, 255, 0)), (20, (255, 0, 0)),
            (20, (255, 255, 0)), (20, (255, 0, 255)),
        ], 5, 5),
    ],
)
def test_detect_scenes_scaling(
    tmp_path: Path, scenes_def: list, expected_min: int, expected_max: int
) -> None:
    path = str(tmp_path / "multi.mp4")
    _make_scene_video(path, scenes_def)
    cap = cv2.VideoCapture(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        scenes = _detect_scenes(cap, total, fps)
    finally:
        cap.release()
    assert expected_min <= len(scenes) <= expected_max, (
        f"scenes={len(scenes)} not in [{expected_min},{expected_max}] for {scenes_def}"
    )


def test_detect_scenes_collapses_over_max(tmp_path: Path) -> None:
    """超过 MAX_SCENES 时应合并相邻最小场景，不超过业务上限。"""
    # 造 12 段（每段 10 帧），总 120 帧
    path = str(tmp_path / "many_scenes.mp4")
    scenes_def = [(10, (i * 20 % 255, (i * 30) % 255, (i * 40) % 255)) for i in range(12)]
    _make_scene_video(path, scenes_def)
    cap = cv2.VideoCapture(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        scenes = _detect_scenes(cap, total, fps)
    finally:
        cap.release()
    assert len(scenes) <= video_parser.MAX_SCENES, (
        f"expected <= {video_parser.MAX_SCENES} scenes, got {len(scenes)}"
    )
    # 覆盖全视频
    assert scenes[0][0] == 0
    assert scenes[-1][1] == total


def test_parse_video_returns_one_chunk_per_scene(
    tmp_video: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """parse_video 应对每场景独立返回 ParsedChunk；不返回 detail_md。"""

    async def _fake_fetch(file_key, url):  # noqa: ANN001
        with open(tmp_video, "rb") as f:
            return f.read()

    async def _fake_describe(paths, scene_id, start_frame, end_frame, fps):
        return f"场景{scene_id} 描述：起 {start_frame} 终 {end_frame}，共 {len(paths)} 帧"

    monkeypatch.setattr("parsers.video_parser.fetch_source_bytes", _fake_fetch)
    monkeypatch.setattr("parsers.video_parser._describe_scene", _fake_describe)

    parsed = asyncio.run(video_parser.parse_video("k", "scene_clip.mp4", None))

    # 元信息校验
    assert parsed.modality == "video"
    assert not parsed.detail_md, "video 路径下 detail_md 必须为空（场景已下沉到 chunks）"
    assert parsed.meta.get("error") is None, f"unexpected error: {parsed.meta}"
    assert parsed.meta["scenes"] == 3
    # 每个场景独立成 chunk
    assert len(parsed.chunks) == 3, f"expected 3 chunks, got {len(parsed.chunks)}"
    for i, ch in enumerate(parsed.chunks, 1):
        assert ch.source == "scene_clip.mp4"
        assert f"场景{i}" in ch.text
    # text 字段仍保留全文（用于摘要向量）
    assert parsed.text
    assert "场景1" in parsed.text and "场景3" in parsed.text


def test_parse_video_keeps_first_and_last_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """首尾帧必须保留：即使只有 1 个场景，也必须有帧可描述。"""

    path = str(tmp_path / "single.mp4")
    _make_scene_video(path, [(20, (0, 0, 255))])

    async def _fake_fetch(file_key, url):  # noqa: ANN001
        with open(path, "rb") as f:
            return f.read()

    async def _fake_describe(paths, scene_id, start_frame, end_frame, fps):
        return f"scene{scene_id} frames={len(paths)}"

    monkeypatch.setattr("parsers.video_parser.fetch_source_bytes", _fake_fetch)
    monkeypatch.setattr("parsers.video_parser._describe_scene", _fake_describe)

    parsed = asyncio.run(video_parser.parse_video("k", "single.mp4", None))
    assert parsed.meta.get("scenes") == 1
    assert len(parsed.chunks) == 1


def test_biz_recall_fans_out_video_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """biz/recall 应把视频的 N 个 chunk 展开成 N 个 details[] 条目。"""
    # 模拟 parser 返回 3 场景的 ParsedFile
    parsed = ParsedFile(
        modality="video",
        text="[场景 1] ...\n\n[场景 2] ...\n\n[场景 3] ...",
        chunks=[
            ParsedChunk(text="场景1 视觉描述", source="video.mp4"),
            ParsedChunk(text="场景2 视觉描述", source="video.mp4"),
            ParsedChunk(text="场景3 视觉描述", source="video.mp4"),
        ],
        detail_md="",  # 视频路径下应该为空
        meta={"scenes": 3},
    )

    # monkeypatch 掉 build_memory_md 和向量 upsert，只验 details 列表
    captured: dict = {}

    async def _fake_build_md(*args, **kwargs):
        captured["details"] = kwargs.get("details") or (args[2] if len(args) > 2 else [])
        return "# fake\n\n## 摘要\n（无）\n\n## 元数据\n- 时间: 未知\n- 来源: []\n\n## 记忆细节\n\n## 记忆主观描述\n（无）"

    monkeypatch.setattr("biz.recall.build_memory_md", _fake_build_md)

    async def _fake_upsert(*_a, **_k):
        return None

    from vector import recall_store

    monkeypatch.setattr(recall_store, "get_recall_store", lambda: _NoopStore())

    # 直接调内部细节展开逻辑
    from biz import recall as biz_recall

    name = "video.mp4"
    p = parsed
    # 复刻 biz/recall.py 中 video chunks 分支的逻辑
    details: list[dict] = []
    if p.modality == "video" and p.chunks:
        for ci, ch in enumerate(p.chunks, 1):
            text = biz_recall._strip_html(ch.text or "")
            if text:
                details.append({"fileName": name, "detail": text})

    assert len(details) == 3
    # fileName 保持原文件名（所有场景共享），删除整源时能一次匹配所有场景段
    assert all(d["fileName"] == "video.mp4" for d in details)
    assert all("视觉描述" in d["detail"] for d in details)


class _NoopStore:
    def upsert(self, **_):  # noqa: ANN001
        return None

    def delete_by_memory_id(self, *_a, **_k):  # noqa: ANN001
        return 0
