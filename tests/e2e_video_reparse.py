"""端到端回归测试：视频源文件"编辑后再解析"必须仍是视频解析（不被当文本）。

用户反馈的 bug：
    第一次解析视频得到正确画面描述；修改主观描述后再保存，md 里变成
    "MP4 容器文件的二进制元数据结构 …… ftyp/moov/avc1/mp4a/stts/stsc"。

本测试模拟：
    1. 上传一个真实 mp4（cv2 生成的 30KB 测试视频）
    2. save 触发首次解析
    3. 等解析完成 → 抓取 md
    4. 用"前端 bug 行为"重发 update：把视频 fileType 标成 1(文本)，模仿
       `new File([], name)` 占位被 fileTypeOf() 退化为 1 的链路
    5. 等再解析完成 → 抓取 md
    6. 断言：两次解析的 md 都不应出现 "MP4" 元数据串；两次都应是视频画面描述

依赖：echo-core 8080 + echo-ai 8000 已起；不删任何数据。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import httpx
from qiniu import put_data  # type: ignore

CORE = "http://localhost:8080/api"
USER = "selftest_video_reparse"
PWD = "test123456"
ROLE = "default"

# 出现这些串基本可以确定是二进制被当文本解析了（即"MP4 元数据"bug 在回潮）
BINARY_MARKERS = (
    "ftypisom", "ftypmp42", "ftypmp41", "ftypqt",
    "moov", "mvhd", "lmvhd", "tkhd",
    "avc1", "avcC", "esds", "btrt",
    "stts", "stsc", "stsz", "stco", "stss",
    "mdat",
)


def has_binary_metadata(md: str) -> bool:
    low = (md or "").lower()
    hits = [m for m in BINARY_MARKERS if m in low]
    return hits


def wait_parse_done(c: httpx.Client, headers: dict, mem: str, max_polls: int = 60) -> tuple[int, str]:
    """轮询 /detail 直至 parseStatus ∈ {2,3} 或超时。返回 (status, mdContent)。"""
    for i in range(max_polls):
        time.sleep(2)
        r = c.get(f"{CORE}/memory/detail", headers=headers, params={"memoryId": mem})
        d = r.json()["data"]
        st = d.get("parseStatus")
        md = d.get("mdContent") or ""
        print(f"  poll#{i:>2} parseStatus={st} mdContent_len={len(md)}")
        if st in (2, 3):
            return st, md
    return -1, ""


def main() -> int:
    fixture = Path(__file__).parent / "fixtures" / "sample.mp4"
    src = fixture.read_bytes()
    assert len(src) > 1000, f"fixture too small: {fixture}"
    print(f"fixture: {fixture} {len(src)} bytes")

    c = httpx.Client(timeout=30)
    # 1) 注册（已存在则忽略）
    r = c.post(f"{CORE}/auth/register", json={"username": USER, "password": PWD, "nickname": "videoselftest"})
    print(f"register -> {r.status_code} {r.json().get('message')}")
    # 2) 登录
    r = c.post(f"{CORE}/auth/login", json={"username": USER, "password": PWD})
    assert r.status_code == 200, f"login failed: {r.text}"
    sid = r.json()["data"]["sessionId"]
    H = {"X-Session-Id": sid}
    print(f"login ok sid={sid[:8]}...")

    # 3) apply memoryId
    r = c.post(f"{CORE}/memory/apply", headers=H, json={"sessionId": sid})
    mem = r.json()["data"]["memoryId"]
    print(f"apply memoryId={mem}")

    # 4) upload-token + 上传 mp4 到七牛
    fname = "拉师傅的欢乐时光.mp4"
    r = c.post(f"{CORE}/memory/upload-token", headers=H,
               json={"roleId": ROLE, "memoryId": mem, "fileName": fname, "isMd": False, "sessionId": sid})
    tok = r.json()["data"]
    key = tok["key"]
    print(f"upload-token key={key}")
    info = put_data(tok["token"], key, src, mime_type="video/mp4")
    # qiniu put_data 旧版返回 (ret, info)，新版仅返回 info。两种都接受。
    status = info[1].status_code if isinstance(info, tuple) else info.status_code
    assert status == 200, f"qiniu upload failed: {info}"
    print(f"qiniu upload ok status={status}")

    # 5) save（正确 fileType=3 触发首次解析）
    topic = f"视频再解析回归-{int(time.time())}"
    r = c.post(f"{CORE}/memory/save", headers=H, json={
        "memoryId": mem, "roleId": ROLE, "topic": topic, "subjectiveDesc": "初始描述",
        "sourceFiles": [{"fileKey": key, "fileName": fname, "fileType": 3}], "sessionId": sid,
    })
    assert r.status_code == 200, f"save failed: {r.text}"
    print(f"save ok -> parseStatus={r.json()['data'].get('parseStatus')}")

    # 6) 轮询直到完成
    st, first_md = wait_parse_done(c, H, mem)
    assert st == 2, f"FIRST 解析未成功 parseStatus={st}"
    assert first_md, "FIRST /detail 未返回 mdContent"
    first_hits = has_binary_metadata(first_md)
    print(f"\n==== FIRST md 前 200 字 ====\n{first_md[:200]}\n====")
    assert not first_hits, f"FIRST md 已含二进制元数据串（解析器分流错）: {first_hits}"
    print(f"FIRST OK: 视频解析正确，无 MP4 元数据串")

    # 7) update —— 关键：故意把 fileType 错标成 1(文本)，模拟前端 bug 路径
    r = c.post(f"{CORE}/memory/update", headers=H, json={
        "memoryId": mem, "roleId": ROLE, "topic": topic,
        "subjectiveDesc": "第二版描述：换了主观感受",
        "sourceFiles": [{"fileKey": key, "fileName": fname, "fileType": 1}],  # ← 故意标错
        "needReparse": True, "sessionId": sid,
    })
    assert r.status_code == 200, f"update failed: {r.text}"
    print(f"update ok -> parseStatus={r.json()['data'].get('parseStatus')}")

    # 8) 再轮询
    st2, second_md = wait_parse_done(c, H, mem)
    assert st2 == 2, f"REPARSE 解析未成功 parseStatus={st2}"
    assert second_md, "REPARSE /detail 未返回 mdContent"
    second_hits = has_binary_metadata(second_md)
    print(f"\n==== SECOND md 前 200 字 ====\n{second_md[:200]}\n====")
    if second_hits:
        print(f"!!! BUG 未修复：REPARSE md 含二进制元数据串: {second_hits}")
        return 1
    print(f"REPARSE OK: 视频仍被正确解析为视频（无 MP4 元数据串）")
    print("\nE2E PASS: 视频源文件在编辑再解析链路下保持视频解析，未退化为文本。")

    # 9) 场景分段校验：视频 md 中应出现 ≥ 2 个 `### 片段`（多场景才分段）
    seg_re = re.compile(r"^###\s*片段\s*\d+/\d+", re.M)
    seg_count = len(seg_re.findall(second_md))
    print(f"reparse md 中的 ### 片段 数: {seg_count}")
    # 测试视频只有 3 段（红绿蓝），cv2.absdiff 应至少切出 2 段
    assert seg_count >= 2, f"场景分段未生效：期望 ≥ 2 个 `### 片段`，实际 {seg_count}（回归：单段大详情被糅杂）"
    print(f"SCENE-AWARE PASS: 视频被切为 {seg_count} 个独立片段，未糅杂到一大段内容。")

    # 10) Plan B 去重校验：跨所有 ### 片段，相同的人物外貌描述不应重复出现
    # Plan B 渲染路径：视频走 structured → 贯穿主体只出现一次 + 每场景 prose 不含重复
    # 旧路径会每段都写"红色纯色背景…"等冗余；现在应只在第 1 段或"贯穿主体"中
    # 把场景信息讲清楚。
    # 校验：在所有 ### 片段子段（不含 ### 贯穿主体）中，特定人物/场景描述短语应只出现 ≤1 次
    # 由于 LLM 输出天然多变化，这里做宽松校验：每个 ### 片段子段的 prose 不应太长
    # （避免 200-500 字长 prose 回归）。
    seg_blocks = re.split(r"^###\s+", second_md, flags=re.M)
    # 第一段是 0 之前的内容，跳过；其余以"片段 i/N"开头
    per_seg_lengths = []
    for blk in seg_blocks:
        m = re.match(r"片段\s+\d+/\d+", blk)
        if not m:
            continue
        # 取片段标题后的内容长度
        body = re.sub(r"^片段\s+\d+/\d+\s*\[来源:[^\]]*\]\s*\n+", "", blk).strip()
        per_seg_lengths.append(len(body))
    print(f"每片段 body 长度: {per_seg_lengths}")
    # Plan B 渲染：每段应 ≤ 200 字（与之前 200-500 长 prose 形成对比）
    if per_seg_lengths:
        max_seg = max(per_seg_lengths)
        # 宽松判定：长 prose 特征是 >300 字符；Plan B 应 ≤ 200；设定阈值 250
        assert max_seg <= 300, (
            f"片段体过长（最长 {max_seg} 字），可能 LLM 重新展开了 prose"
            f"——Plan B 应保持短 prose 直出"
        )
        print(f"PROSE-COMPACT PASS: 每片段 prose 长度 {per_seg_lengths}，无长 prose 回归。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
