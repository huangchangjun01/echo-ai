"""回归测试：视频源文件在"编辑后重新解析"时被当成文本解析的缺陷。

缺陷现象（用户反馈）：
    第一次解析视频得到正确的画面描述；修改主观描述后再次保存，md 里的片段变成了
    "MP4 容器文件的二进制元数据结构 …… ftyp/moov/avc1/mp4a/stts/stsc"。

根因链路：
    echo-web 编辑态用 `new File([], fileName)` 占位（File.type 为空串），保存时按
    mime 重新推导 fileType 得到 1(文本) → echo-core 透传 → registry.parse_file
    分派到 parse_text → 把 mp4 原始字节 decode 成文本喂给 LLM。

本文件覆盖 echo-ai 侧的两道防线：
    1. registry：扩展名与 fileType 冲突时以扩展名为准（mp4 一定不是文本）。
    2. text_parser：内容明显是二进制时直接判失败，绝不把乱码喂给 LLM。
"""

from __future__ import annotations

import pytest

from parsers import registry
from parsers.text_parser import parse_text

# 一段真实 MP4 头部：ftyp box + moov/mvhd 片段。
# 用 utf-8 errors="replace" 解码后正是用户看到的 "ftyp isom iso2 avc1 mp41 … lmvhd" 文本。
MP4_HEADER = (
    b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
    b"\x00\x00\x00\x08free\x00\x00\x1f\x54moov\x00\x00\x00lmvhd"
    b"\x00\x00\x00\x00\xc7\x21\x5c\x8d\x00\x00\x00\x00\x00\x00\x03\xe8"
)


class _Recorder:
    """记录 registry 最终分派到了哪个解析器。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def make(self, name: str):
        async def _fn(file_key: str, file_name: str, url: str | None):
            self.calls.append(name)
            from parsers.base import ParsedFile

            return ParsedFile(modality=name, text="ok", detail_md="ok")

        return _fn


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    for mod in ("parse_text", "parse_image", "parse_video", "parse_audio"):
        monkeypatch.setattr(registry, mod, rec.make(mod.replace("parse_", "")))
    return rec


# ===== 防线 1：registry 扩展名兜底 =====


async def test_mp4_with_wrong_filetype_routes_to_video(recorder: _Recorder) -> None:
    """核心回归：fileType 误传 1(文本)，但文件名是 .mp4 → 必须走视频解析器。"""
    await registry.parse_file("memory/u/r/m/abc.mp4", "拉师傅的欢乐时光.mp4", 1, None)
    assert recorder.calls == ["video"], "mp4 被误当作文本解析（用户反馈的 MP4 元数据 bug）"


@pytest.mark.parametrize(
    ("file_name", "wrong_type", "expected"),
    [
        ("clip.mov", 1, "video"),
        ("clip.mkv", 2, "video"),
        ("song.mp3", 1, "audio"),
        ("voice.wav", 3, "audio"),
        ("photo.jpg", 1, "image"),
        ("photo.png", 4, "image"),
        ("note.txt", 3, "text"),
        ("readme.md", 2, "text"),
    ],
)
async def test_extension_overrides_wrong_filetype(
    recorder: _Recorder, file_name: str, wrong_type: int, expected: str
) -> None:
    """任意模态：扩展名可识别时，以扩展名为准。"""
    await registry.parse_file("k", file_name, wrong_type, None)
    assert recorder.calls == [expected]


async def test_unknown_extension_falls_back_to_declared_type(recorder: _Recorder) -> None:
    """扩展名无法识别时，仍然尊重上游声明的 fileType（不能反向破坏正常链路）。"""
    await registry.parse_file("k", "backup.bin", 3, None)
    assert recorder.calls == ["video"]


async def test_correct_filetype_is_untouched(recorder: _Recorder) -> None:
    """扩展名与 fileType 一致时，分派结果不变。"""
    await registry.parse_file("k", "clip.mp4", 3, None)
    assert recorder.calls == ["video"]


async def test_unknown_type_and_unknown_extension_still_raises(recorder: _Recorder) -> None:
    """两边都无法判定时维持既有语义：抛 ValueError，由上层记为解析失败。"""
    with pytest.raises(ValueError):
        await registry.parse_file("k", "backup.bin", 9, None)


# ===== 防线 2：text_parser 二进制内容拒绝 =====


async def test_parse_text_rejects_binary_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """即使真的走到 parse_text，二进制内容也必须判失败，而不是把乱码写进 md。"""

    async def _fake_fetch(file_key, url):  # noqa: ANN001
        return MP4_HEADER

    monkeypatch.setattr("parsers.text_parser.fetch_source_bytes", _fake_fetch)

    parsed = await parse_text("k", "拉师傅的欢乐时光.mp4", None)

    assert parsed.meta.get("error"), "二进制内容必须判为解析失败"
    assert not parsed.detail_md, "二进制内容绝不能进入 detail_md（会被 LLM 当正文总结）"
    assert "ftyp" not in parsed.text


async def test_parse_text_accepts_normal_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    """正常中英文文本不受影响（防止兜底逻辑误伤）。"""

    async def _fake_fetch(file_key, url):  # noqa: ANN001
        return "今天和拉师傅一起吃了火锅，很开心。\nHello, world!\n".encode()

    monkeypatch.setattr("parsers.text_parser.fetch_source_bytes", _fake_fetch)

    parsed = await parse_text("k", "note.txt", None)

    assert not parsed.meta.get("error")
    assert "拉师傅" in parsed.detail_md
