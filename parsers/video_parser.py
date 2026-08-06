"""视频解析器：按场景自适应抽帧 → 视觉模型逐场景描述 → 每场景一段记忆。

设计要点：
- 用 cv2.absdiff 检测帧间变化（基于降采样灰度帧），自动切场景；
- 每个场景内抽 1~3 张代表帧（前/中/后），避免 LLM 拿到相似帧浪费 token；
- 每个场景独立一次 LLM 调用，输出"## 场景 N (t=~Xs) ..."作为单条 ParsedChunk；
- 视频的 ParsedFile.chunks 字段承载 N 个场景；biz/recall 看到 chunks 非空时
  会展开成 N 条 details[] 记录，自然生成 N 个 `### 片段` 子段；
- 严格首尾帧保留（即使首尾属于同一场景也保留），保证时间覆盖不漏掉起止。

为何不用 PySceneDetect / ffmpeg：当前环境未装；cv2 已能完成 diff + 抽帧，
且 diff 法足够应对"用户自拍、风景、生活记录"这类典型场景切分。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile

import cv2  # type: ignore

from llm.client import get_llm_client
from utils.downloader import fetch_source_bytes
from utils.request_context import log_exception, merge_extra
from utils.temp_files import safe_remove
from .base import ParsedFile, ParsedChunk

logger = logging.getLogger("echo-ai.parsers")

# —— 抽帧与场景切分参数 —— #
MAX_KEYFRAMES = 8                       # 旧版硬上限，现仅作为"总抽帧数兜底"参考
MAX_SCENES = 8                          # 单视频最多切多少个场景（业务上限，防滥用）
MIN_SCENE_FRAMES = 8                    # 一个场景至少多少帧（避免切太碎）
SCENE_DIFF_THRESHOLD = 22.0             # 帧间平均绝对差（0~255）超过此值视为场景切换
SCENE_PROBE_W = 64                      # 用于 diff 比对的降采样宽度
SCENE_PROBE_H = 48                      # 用于 diff 比对的降采样高度
FRAMES_PER_SCENE = (1, 3)               # 每个场景采 1~3 张代表帧

# —— 单帧 / 单场景视觉描述 prompt（结构化 + prose 双段）—— #
# Stage 1 LLM 输出预期：
#   【第一段】8 字段结构化 markdown（用于跨场景聚合去重）
#   【空行】
#   【第二段】1~2 句连贯中文 prose（直接作为最终片段渲染内容）
_KEYFRAME_PROMPT = (
    "你是视频场景结构化抽取助手。从提供的 1~3 张同场景代表帧中提取信息。\n"
    "\n"
    "【第一步】输出 8 字段结构化数据，每行以 `- 字段名: ` 开头，"
    "用全角或半角冒号均可。若该字段无内容写「无」，不要任何前后缀解释。\n"
    "字段说明：\n"
    "  - 人物: 主体身份(外貌, 服饰)，多个人物用「、」分隔\n"
    "  - 场景: 物理环境（地点+时间+氛围），10~25 字\n"
    "  - 动作: 主体+动作，多个用「、」分隔\n"
    "  - 物品: 物品(状态)，多个用「、」分隔\n"
    "  - 文字: 画面可见文字（若无写「无」）\n"
    "  - 表情: 主体+表情，多个用「、」分隔\n"
    "  - 颜色方位: 整体色调+主体在画面中的位置，10~25 字\n"
    "  - 变化: 与上一场景相比的变化（首场景写「初始」）\n"
    "\n"
    "【第二步】紧接着（与第一步之间用空行分隔）写 1~2 句中文连贯描述本场景：\n"
    "约 60~120 字。直接写散文，描述本场景的所见所闻，"
    "不要任何标题/前缀/标签/换行。\n"
)

# 单场景 detail 长度上限：每场景 1500 字内足够，N 场景 × 1500 = 总长度可控
# （5 场景 ≈ 7500 字符，仍在小模型上下文窗口内）
_MAX_PER_SCENE_CHARS = 1500

# MiniMax-M3 类推理模型会把 chain-of-thought 放在  (html]> 块里输出，
# 必须在此层先剥离。否则 build_memory_md 的小模型会被半截思考污染。
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.M)

_FENCE_RE = re.compile(r"^\s*```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$", re.M)


def _strip_think_and_fences(text: str) -> str:
    """清理 LLM 视觉输出里的非正文：链式推理块 / 代码围栏。"""
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1)
    return text.lstrip("\n\r\t ")


def _detect_scenes(
    cap: cv2.VideoCapture,
    total_frames: int,
    fps: float,
) -> list[tuple[int, int]]:
    """基于帧间 absdiff 检测场景边界。

    返回 `[(start, end_exclusive), ...]`，每个区间至少 MIN_SCENE_FRAMES 帧。
    算法：
      1. 降采样到 SCENE_PROBE_W x SCENE_PROBE_H 灰度图，间隔 STEP 帧采一次
         （步进由总帧数 / 30 估算，目标 ~30 个探针帧即可覆盖大多数视频）。
      2. 探针帧两两 absdiff → 平均绝对差，与 SCENE_DIFF_THRESHOLD 比较。
      3. 标记所有超过阈值的"探针位置"为候选切分点。
      4. 后处理：把候选切分点投影回原始帧号；保证每个区间 ≥ MIN_SCENE_FRAMES。
      5. 强制保留首尾边界。
    """
    if total_frames <= 0 or fps <= 0:
        return [(0, max(1, total_frames))]

    # 1) 探针步进：目标 30 个探针，但单场景至少 MIN_SCENE_FRAMES 帧
    n_probes_target = 30
    step_probe = max(MIN_SCENE_FRAMES, total_frames // n_probes_target)
    probe_indices: list[int] = list(range(0, total_frames, step_probe))
    if probe_indices[-1] != total_frames - 1:
        probe_indices.append(total_frames - 1)

    # 2) 读出所有探针帧（小尺寸灰度）
    prev_gray: cv2.typing.MatLike | None = None
    diffs: list[float] = []
    probe_frames: dict[int, cv2.typing.MatLike] = {}
    for idx in probe_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            diffs.append(0.0)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (SCENE_PROBE_W, SCENE_PROBE_H), interpolation=cv2.INTER_AREA)
        probe_frames[idx] = frame  # 全尺寸保留，稍后按场景回采
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = float(diff.mean())
            diffs.append(mean_diff)
        else:
            diffs.append(0.0)
        prev_gray = gray

    # 3) 把超阈值的探针位置标为切分点
    candidates: list[int] = [0]
    for i, d in enumerate(diffs[1:], start=1):
        if d >= SCENE_DIFF_THRESHOLD:
            candidates.append(probe_indices[i])
    if candidates[-1] != total_frames - 1:
        candidates.append(total_frames)

    # 4) 合并过近的切分点（保证每个区间 ≥ MIN_SCENE_FRAMES）
    merged: list[int] = [candidates[0]]
    for c in candidates[1:]:
        if c - merged[-1] >= MIN_SCENE_FRAMES:
            merged.append(c)
        # 否则跳过这个切分点（与上一个合并）
    if merged[-1] != total_frames - 1:
        # 末段太短：并入前一段
        if len(merged) >= 2:
            merged.pop()
        merged.append(total_frames)

    # 5) 转成区间
    scenes: list[tuple[int, int]] = []
    for i in range(len(merged) - 1):
        s = merged[i]
        e = merged[i + 1]
        if e - s < 1:
            e = s + 1
        scenes.append((s, e))

    # 6) 业务上限截断：合并相邻最小场景直到 ≤ MAX_SCENES
    while len(scenes) > MAX_SCENES:
        # 找出长度最小的一段，与其前一段合并
        lens = [e - s for s, e in scenes]
        i_min = lens.index(min(lens))
        i_merge = max(0, i_min - 1)
        s, _ = scenes[i_merge]
        _, e = scenes[i_merge + 1]
        scenes[i_merge] = (s, e)
        del scenes[i_merge + 1]

    return scenes


def _sample_scene_frames(
    cap: cv2.VideoCapture,
    scene: tuple[int, int],
    out_dir: str,
    scene_id: int,
) -> list[tuple[int, str]]:
    """在一个场景 [s, e) 内采 1~3 张代表帧，返回 [(frame_idx, jpg_path), ...]。

    策略：场景 ≥ 24 帧采 3 张（前/中/后），≥ 8 帧采 2 张（前/后），否则采 1 张（中点）。
    """
    s, e = scene
    length = e - s
    if length <= 0:
        return []

    n = 1
    if length >= 24:
        n = 3
    elif length >= FRAMES_PER_SCENE[0] * 4:
        n = 2

    if n == 1:
        offsets = [(s + length // 2,)]
    elif n == 2:
        offsets = [(s,), (e - 1,)]
    else:
        offsets = [(s,), (s + length // 2,), (e - 1,)]

    out: list[tuple[int, str]] = []
    for k, (frame_idx,) in enumerate(offsets):
        if frame_idx < s or frame_idx >= e:
            frame_idx = min(max(frame_idx, s), e - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        p = os.path.join(out_dir, f"scene{scene_id:02d}_k{k}.jpg")
        cv2.imwrite(p, frame)
        out.append((frame_idx, p))
    return out


async def _describe_scene(
    scene_paths: list[str],
    scene_id: int,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> str:
    """把一个场景的代表帧打包成多模态 prompt 调一次 LLM，返回「结构化字段 + prose」双段。

    输入：同场景 1~3 张 jpg 路径。
    输出：清理过 think/fence 的双段 markdown：
        - 字段: 值
        ...

        <1~2 句连贯 prose>

    解析失败（LLM 漂移）→ 返回空串，由调用方决定走 fallback。
    """
    import base64

    images: list[dict] = []
    for p in scene_paths:
        with open(p, "rb") as f:
            b = base64.b64encode(f.read()).decode("ascii")
        images.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b}"},
            }
        )

    if fps > 0:
        t_start = start_frame / fps
        t_end = end_frame / fps
        time_hint = f"约 {t_start:.1f}s ~ {t_end:.1f}s"
    else:
        time_hint = ""

    user_text = (
        f"这是视频中的第 {scene_id} 个场景（{time_hint}）。"
        f"包含 {len(scene_paths)} 张同场景代表帧。请按系统要求先输出 8 字段结构化数据，"
        f"空行后再写 1~2 句连贯描述。"
    )

    client = get_llm_client()
    messages = [
        {"role": "system", "content": _KEYFRAME_PROMPT},
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}, *images],
        },
    ]
    try:
        # 结构化字段多（8 行）+ prose（~100 字）→ 提升 max_tokens
        resp = await client.chat(messages, max_tokens=800, temperature=0.4)
        try:
            raw = (resp["choices"][0]["message"]["content"] or "").strip()
            return _strip_think_and_fences(raw)
        except Exception:
            return ""
    except Exception as e:
        log_exception(
            logger,
            "scene vision failed",
            exc=e,
            level=logging.WARNING,
            stage="parser_video",
            event="scene_vision_error",
            scene_id=scene_id,
        )
        return ""


async def parse_video(file_key: str, file_name: str, url: str | None) -> ParsedFile:
    """视频解析主流程：抽帧 → 场景切分 → 每场景独立视觉 LLM → N 个 ParsedChunk。

    返回 ParsedFile 的关键字段：
      - chunks  : 每个场景一个 ParsedChunk，text=场景描述，source=file_name
      - text    : 全部场景描述拼接（用于摘要向量嵌入；不进入 md 渲染）
      - detail_md: 空串（场景已下沉到 chunks，由 biz/recall 展开成 N 个片段）
      - meta    : {scenes, frames, bytes, fps}
    """
    if not file_key and not url:
        return ParsedFile(modality="video", meta={"error": "no_key_or_url"})

    workdir: str | None = None
    try:
        raw = await fetch_source_bytes(file_key or None, url or None)
        workdir = tempfile.mkdtemp(prefix="echo-video-")
        video_path = os.path.join(workdir, "src.mp4")
        with open(video_path, "wb") as f:
            f.write(raw)

        # 抽帧 / 场景切分（同步，丢线程池）
        scenes, fps, total, frames_meta = await asyncio.to_thread(
            _extract_scenes, video_path, workdir
        )
        if not scenes:
            return ParsedFile(modality="video", meta={"error": "no_scenes"})

        # 每个场景：采帧 → 视觉描述 → 解析为 SceneExtraction → 设置 chunk.meta
        from .scene_aggregator import parse_scene_extraction  # 避免循环依赖

        per_scene_texts: list[str] = []
        per_scene_chunks: list[ParsedChunk] = []
        for i, ((s, e), paths_with_idx) in enumerate(zip(scenes, frames_meta), 1):
            paths = [p for _, p in paths_with_idx]
            if not paths:
                continue
            d = await _describe_scene(paths, scene_id=i, start_frame=s, end_frame=e, fps=fps)
            if not d:
                # 视觉失败：给一段"场景{i} 无可用描述"占位，避免下游把整个视频判定为失败
                d = f"（场景 {i}：视觉识别失败）"

            # 解析为 SceneExtraction。失败时给空 dict（meta 仍存在但没 structured 字段）
            ext = parse_scene_extraction(d, idx=i, source=file_name)
            structured_meta: dict = {}
            if ext is not None:
                structured_meta = {
                    "structured": True,
                    "characters": ext.characters,
                    "setting": ext.setting,
                    "actions": ext.actions,
                    "objects": ext.objects,
                    "text": ext.text,
                    "expressions": ext.expressions,
                    "color_spatial": ext.color_spatial,
                    "deltas": ext.deltas,
                    "prose": ext.prose,
                }
                # 渲染用文本 = 优先 prose，否则退化 raw（解析失败时）
                render_text = ext.prose or d
            else:
                structured_meta = {"structured": False, "prose": d}
                render_text = d

            # 单场景长度上限（截断时优先保留 prose 完整）
            if len(render_text) > _MAX_PER_SCENE_CHARS:
                render_text = render_text[:_MAX_PER_SCENE_CHARS]

            per_scene_texts.append(f"[场景 {i}] {render_text}")
            per_scene_chunks.append(
                ParsedChunk(text=render_text, source=file_name, meta=structured_meta)
            )

        if not per_scene_chunks:
            return ParsedFile(modality="video", meta={"error": "all_scenes_failed"})

        logger.info(
            "video parsed (scene-aware, structured)",
            extra=merge_extra(
                stage="parser_video",
                event="ok",
                file_name=file_name,
                scenes=len(per_scene_chunks),
                fps=fps,
                total_frames=total,
                bytes=len(raw),
            ),
        )

        # chunks 承载 N 个场景（下游 biz/recall 会展开成 N 个 details[] 条目）。
        # text 字段保留完整描述供摘要向量；detail_md 留空以避免与 chunks 重复。
        return ParsedFile(
            modality="video",
            text="\n\n".join(per_scene_texts)[:3000],
            chunks=per_scene_chunks,
            detail_md="",  # 显式置空：场景细节已通过 chunks 走 N 段渲染
            meta={
                "scenes": len(per_scene_chunks),
                "frames": sum(len(p) for p in frames_meta),
                "bytes": len(raw),
                "fps": round(fps, 2),
            },
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


def _extract_scenes(
    video_path: str, out_dir: str
) -> tuple[list[tuple[int, int]], float, int, list[list[tuple[int, str]]]]:
    """同步封装：开 cap → 切场景 → 采代表帧。返回 (scenes, fps, total, per_scene_frames)。

    per_scene_frames[i] = 该场景采到的 [(frame_idx, jpg_path), ...]。
    """
    cap = cv2.VideoCapture(video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        # 退化：拿不到总帧数时按 1 段处理
        if total <= 0:
            ret, frame = cap.read()
            if not ret:
                return [], fps, 0, []
            p = os.path.join(out_dir, "scene01_k0.jpg")
            cv2.imwrite(p, frame)
            return [(0, 1)], fps, 1, [[(0, p)]]

        # 场景切分
        scenes = _detect_scenes(cap, total, fps)

        # 每个场景采代表帧
        per_scene: list[list[tuple[int, str]]] = []
        for i, scene in enumerate(scenes, 1):
            frames = _sample_scene_frames(cap, scene, out_dir, scene_id=i)
            per_scene.append(frames)
        return scenes, fps, total, per_scene
    finally:
        cap.release()
