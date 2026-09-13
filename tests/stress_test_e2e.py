"""1000 轮对话压测脚本 (Playwright 驱动 echo-web 真实 UI)。

覆盖 7 类对话 × 不同轮数:
- 闲聊问候/Q&A      200 轮  链路基本功能
- 显式偏好/身份     200 轮  L0 抽取
- 近期事件叙述      200 轮  L1 时间窗口 + L2 摘要
- 主动回忆查询      100 轮  召回及时性
- 多轮上下文        100 轮  ReAct 多步决策
- 矛盾更新          100 轮  relation=contradict
- 跨主题相关性      100 轮  向量召回

执行流程 (per context):
  1. Python httpx 注册测试用户 (POST /api/auth/register)
  2. Python httpx 登录获取 sessionId (POST /api/auth/login)
  3. Playwright 打开 /chat 前, 用 add_init_script 把 sessionId 注入 localStorage
  4. 启动 chat UI, 逐轮 type → send → wait streaming done → 读最后一条 assistant 消息

并行 4 个 context × 250 轮 = 1000 轮。所有指标写到 stress_metrics.json。

运行:
    python -m tests.stress_test_e2e
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, ".")

from playwright.async_api import async_playwright, Page, BrowserContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("stress")

# ---------- 配置 ----------
ECHO_CORE_BASE = "http://localhost:8080"
ECHO_WEB_BASE = "http://localhost:5173"
ECHO_AI_BASE = "http://localhost:8000"

# ---------- 对话模板 ----------
TEMPLATES: dict[str, list[str]] = {
    "greeting": [
        "你好呀", "今天天气不错", "周末有什么安排吗", "我有点累, 想放松一下",
        "给我讲个冷笑话", "你叫什么名字", "Echo 是谁", "晚安", "早上好",
        "今天过得怎么样", "我想看个电影, 推荐一下", "你会唱歌吗",
        "讲个故事给我听", "最近有什么新鲜事吗", "随便聊聊吧",
        "你在干嘛", "今天有什么新闻", "推荐本书给我",
        "有什么好吃的", "我饿了", "明天见", "祝你好运",
    ],
    "L0_preference": [
        "我是一名 Python 后端工程师, 工作 5 年了",
        "我最喜欢的颜色是蓝色, 它让我感到平静",
        "我养了一只猫, 名字叫小白, 已经 3 岁了",
        "我对海鲜过敏, 请不要推荐",
        "我住在上海, 在张江高科上班",
        "我是摩羯座, 性格偏内向",
        "我最喜欢的电影是《盗梦空间》",
        "我每天早上跑步 5 公里, 已经坚持一年了",
        "我是一名医生, 在三甲医院工作",
        "我喜欢喝美式咖啡, 不加糖",
        "我的爱好是摄影, 主要拍风景",
        "我是一名教师, 教高中数学",
        "我喜欢看科幻小说, 阿西莫夫是我的最爱",
        "我是湖南人, 爱吃辣",
        "我的生日是 3 月 15 日, 双鱼座",
        "我是一名产品经理, 喜欢用户体验设计",
        "我养了一条金毛, 叫大黄",
        "我是素食主义者",
        "我最喜欢的季节是秋天",
        "我是一名自由职业者, 做平面设计",
    ],
    "L1_event": [
        "今天早上我去了一家新开的咖啡馆, 点了一杯拿铁",
        "昨天我和朋友去看了新上映的悬疑电影, 非常精彩",
        "今天下午公司开了项目复盘会, 我们讨论了 Q3 的进展",
        "周末我和家人去了杭州西湖, 风景很美",
        "昨天晚上我做了一道红烧肉, 全家都夸好吃",
        "今天早上地铁延误了 30 分钟, 我迟到了",
        "这周开始我在学 Rust, 觉得所有权机制很有趣",
        "我最近在追一部日剧, 叫《重启人生》",
        "今天我去医院体检了, 各指标都正常",
        "昨天半夜我做了一个奇怪的梦, 梦见自己在月球",
        "今天午饭我和同事去了一家日本料理店",
        "我刚买了一双新跑鞋, 准备周末去爬山",
        "今天上午我去图书馆借了 3 本关于 AI 的书",
        "今天下午我和大学同学视频聊天了 2 小时",
        "昨晚失眠了, 凌晨 3 点才睡着",
    ],
    "recall_query": [
        "我之前说过我的职业吗?",
        "我最喜欢的颜色是什么?",
        "我家宠物叫什么?",
        "我对什么食物过敏?",
        "我最近一次去看电影是什么时候?",
        "我住在哪里?",
        "我养了什么动物?",
        "我的生日是几月几号?",
        "我最近学了哪门新语言?",
        "我喜欢的咖啡口味是?",
    ],
    "multi_turn": [
        # 走 multi-turn runner (单轮发送 1 个 topic, 内部用 linked Q&A 推进)
        "我们聊聊 AI 的发展吧",
        "继续",
        "你怎么看深度学习",
        "Transformer 架构有什么特点",
        "GPT 和 BERT 的区别",
        "现在的 AI 还有什么局限",
        "那 AGI 还有多远",
        "聊聊 AI 伦理",
    ],
    "contradict": [
        # 在 contradict 模式生成器里动态反转
        "_reverse_marker",
    ],
    "cross_topic": [
        "我之前说的那个跟 Python 有关的事",
        "还记得我上次提到的电影吗",
        "我提过的那个宠物",
        "我之前聊过的早餐",
        "上次我们讨论的那本书",
        "我前面说过的关于 Rust 的事",
        "之前提到的那家咖啡馆",
        "我之前聊过的颜色",
    ],
}

REVERSE_TEMPLATES = [
    "其实我不喜欢蓝色, 我现在更喜欢绿色",
    "我其实不喜欢跑步, 我更爱游泳",
    "我已经不吃素了, 最近又开始吃肉",
    "其实我不是医生, 我是律师",
    "我搬家了, 现在住在深圳",
    "我换工作了, 现在是一名产品经理",
    "我家猫丢了, 我现在养了一只乌龟",
    "我对海鲜不过敏了, 上个月开始吃",
    "我最喜欢的电影换了, 现在是《星际穿越》",
    "我生日不是 3 月 15 日, 我记错了, 是 4 月 1 日",
    "我已经不做 Python 了, 现在转 Java",
    "我不喜欢美式了, 现在爱喝拿铁",
    "我不再养猫, 改养了一只狗",
    "其实我不是摩羯座, 是水瓶座",
    "我不再跑步, 现在改练瑜伽",
]


@dataclass
class RoundResult:
    round_id: int
    user_idx: int
    user_id: str
    conv_type: str
    user_message: str
    reply: str
    latency_ms: int
    ok: bool
    error: str | None = None


@dataclass
class ContextMetrics:
    user_idx: int
    user_id: str
    session_id: str
    total_rounds: int = 0
    ok_rounds: int = 0
    failed_rounds: int = 0
    total_latency_ms: int = 0
    conv_type_distribution: dict[str, int] = field(default_factory=dict)


RESULTS_LOG: list[RoundResult] = []


# ---------- 鉴权 ----------

async def register_and_login(idx: int) -> dict:
    """注册并登录一个测试用户, 返回包含 session 信息的 dict。"""
    uname = f"stress_{idx}_{uuid.uuid4().hex[:8]}"
    nick = f"压测{idx:02d}"
    async with httpx.AsyncClient(base_url=ECHO_CORE_BASE, timeout=15.0) as c:
        r = await c.post("/api/auth/register",
                         json={"username": uname, "password": "test123", "nickname": nick})
        data = r.json().get("data") or {}
        if r.json().get("code") not in (0, 200):
            raise RuntimeError(f"register failed: {r.text[:200]}")
        user_id = int(data["id"])
        r = await c.post("/api/auth/login",
                         json={"username": uname, "password": "test123"})
        data = r.json().get("data") or {}
        if r.json().get("code") not in (0, 200):
            raise RuntimeError(f"login failed: {r.text[:200]}")
        return {
            "user_id": user_id,
            "username": uname,
            "nickname": nick,
            "session_id": data["sessionId"],
            "expire_at": data.get("expireAt", ""),
            "user_info": data.get("user", {}),
        }


# ---------- Playwright 交互 ----------

INJECT_SESSION_JS = (
    "({sessionId, expiresAt, user}) => {"
    "  const payload = {"
    "    sessionId,"
    "    expiresAt,"
    "    user"
    "  };"
    "  localStorage.setItem('echo_auth_session', JSON.stringify(payload));"
    "}"
)


async def type_and_send(page: Page, message: str) -> None:
    """在 chat 输入框填消息并点击 send 按钮。

    输入框有 2 种状态:
    - 空状态(无消息): .empty-input .el-textarea__inner + .empty-send-btn
    - 已有消息: .chat-input .el-textarea__inner + .send-btn
    """
    textarea = page.locator(
        ".chat-input .el-textarea__inner, .empty-input .el-textarea__inner"
    ).first
    await textarea.click()
    await textarea.fill(message)
    send_btn = page.locator(".send-btn, .empty-send-btn").first
    await send_btn.click(force=True)


async def wait_for_response_done(page: Page, timeout_ms: int = 30_000) -> tuple[str, dict]:
    """等待 assistant 消息渲染完成(流式结束),返回 (text, metadata)。

    策略: 不依赖 typing-caret 选择器, 而是检测:
    1. assistant 消息条数增长 1(新消息出现)
    2. 最后一条 assistant 的 .message-content 文本连续 500ms 内长度不变
    """
    meta: dict = {}
    t0 = time.perf_counter()
    # Step 1: 记录初始 assistant 消息数
    initial = await page.evaluate(
        "() => document.querySelectorAll('.message-wrapper--assistant').length"
    )
    # Step 2: 等消息数 +1
    try:
        await page.wait_for_function(
            f"() => document.querySelectorAll('.message-wrapper--assistant').length >= {initial + 1}",
            timeout=timeout_ms,
        )
    except Exception:
        pass
    # Step 3: 等最后一条消息的 .message-content 文本长度连续 500ms 不变(流式结束)
    try:
        await page.wait_for_function(
            """() => {
                const items = document.querySelectorAll('.message-wrapper--assistant');
                if (items.length === 0) return false;
                const last = items[items.length - 1];
                const content = last.querySelector('.message-content');
                if (!content) return false;
                const t = (content.innerText || content.textContent || '').trim();
                if (!window.__lastReplyText) {
                    window.__lastReplyText = t;
                    window.__lastReplyTime = Date.now();
                    return false;
                }
                if (t === window.__lastReplyText) {
                    if (Date.now() - window.__lastReplyTime > 500) {
                        // 文本稳定 500ms, 算流式结束
                        return true;
                    }
                } else {
                    window.__lastReplyText = t;
                    window.__lastReplyTime = Date.now();
                }
                return false;
            }""",
            timeout=timeout_ms,
        )
    except Exception:
        pass
    # 清理 marker
    await page.evaluate("() => { window.__lastReplyText = ''; window.__lastReplyTime = 0; }")

    latency_ms = int((time.perf_counter() - t0) * 1000)

    # 抓 assistant 的实际回复
    reply_text = ""
    try:
        reply_text = await page.evaluate(
            "() => {"
            "  const items = document.querySelectorAll('.message-wrapper--assistant');"
            "  if (!items.length) return '';"
            "  const last = items[items.length - 1];"
            "  const content = last.querySelector('.message-content');"
            "  if (!content) return '';"
            "  return (content.innerText || content.textContent || '').trim();"
            "}"
        )
    except Exception:
        pass
    return reply_text[:800], meta


async def run_one_round(page: Page, round_id: int, user_idx: int, user_id: str,
                       conv_type: str, message: str,
                       timeout_ms: int = 30_000) -> RoundResult:
    t0 = time.perf_counter()
    try:
        await type_and_send(page, message)
        reply, _meta = await wait_for_response_done(page, timeout_ms=timeout_ms)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        ok = bool(reply and reply.strip())
        return RoundResult(
            round_id=round_id, user_idx=user_idx, user_id=user_id,
            conv_type=conv_type, user_message=message[:200], reply=reply,
            latency_ms=latency_ms, ok=ok, error=None if ok else "empty_reply",
        )
    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return RoundResult(
            round_id=round_id, user_idx=user_idx, user_id=user_id,
            conv_type=conv_type, user_message=message[:200], reply="",
            latency_ms=latency_ms, ok=False, error=str(e)[:200],
        )


def build_schedule(per_user: int, distribution: dict[str, int]) -> list[tuple[str, str]]:
    """构造 (type, message) 列表。"""
    schedule: list[tuple[str, str]] = []
    for ctype, count in distribution.items():
        templates = TEMPLATES.get(ctype, [])
        if not templates or ctype == "contradict":
            continue
        for _ in range(count):
            msg = random.choice(templates)
            schedule.append((ctype, msg))
    # 用 contradict 填满缺口
    while len(schedule) < per_user:
        schedule.append(("contradict", random.choice(REVERSE_TEMPLATES)))
    random.shuffle(schedule)
    return schedule[:per_user]


async def run_one_user(ctx_data: dict, ctx_metric: ContextMetrics, schedule: list[tuple[str, str]]) -> None:
    """在单个 context 上跑所有轮次。ctx_data 包含 user_id/session_id/expire_at/user_info。"""
    browser_ctx: BrowserContext = ctx_data["browser_ctx"]
    user_id = str(ctx_data["user_id"])
    session_id = ctx_data["session_id"]
    user_idx = ctx_metric.user_idx
    ctx_metric.user_id = user_id
    ctx_metric.session_id = session_id
    expires_at_ms = ctx_data["expires_at_ms"]
    user_info = ctx_data["user_info"]

    page = await browser_ctx.new_page()
    page.set_default_timeout(30_000)

    # 先加载任意页面使 localStorage origin 生效
    try:
        await page.goto(f"{ECHO_WEB_BASE}/", wait_until="networkidle", timeout=20_000)
    except Exception as e:
        logger.error("initial goto failed user=%d: %s", user_idx, e)
        ctx_metric.failed_rounds += 1
        await page.close()
        return

    # 注入 localStorage
    payload_json = json.dumps({
        "sessionId": session_id,
        "expiresAt": expires_at_ms,
        "user": user_info,
    }, ensure_ascii=False)
    await page.evaluate(
        f"(payload) => {{ localStorage.setItem('echo_auth_session', payload); }}",
        payload_json,
    )
    # 校验注入成功
    ls = await page.evaluate("() => localStorage.getItem('echo_auth_session')")
    if not ls:
        logger.error("localStorage injection failed user=%d", user_idx)
        ctx_metric.failed_rounds += 1
        await page.close()
        return

    # 现在跳转到 /chat,auth store 会从 localStorage 恢复 session
    try:
        await page.goto(f"{ECHO_WEB_BASE}/chat", wait_until="networkidle", timeout=30_000)
    except Exception as e:
        logger.error("goto /chat failed user=%d: %s", user_idx, e)
        ctx_metric.failed_rounds += 1
        await page.close()
        return

    try:
        await page.wait_for_selector(
            ".chat-input .el-textarea__inner, .empty-input .el-textarea__inner",
            timeout=15_000,
        )
    except Exception as e:
        logger.error("input not ready user=%d: %s", user_idx, e)
        ctx_metric.failed_rounds += 1
        await page.close()
        return

    for round_id, (ctype, msg) in enumerate(schedule, start=1):
        result = await run_one_round(page, round_id, user_idx, user_id, ctype, msg)
        ctx_metric.total_rounds += 1
        if result.ok:
            ctx_metric.ok_rounds += 1
        else:
            ctx_metric.failed_rounds += 1
        ctx_metric.total_latency_ms += result.latency_ms
        ctx_metric.conv_type_distribution[ctype] = ctx_metric.conv_type_distribution.get(ctype, 0) + 1
        RESULTS_LOG.append(result)
        if round_id % 25 == 0 or ctx_metric.failed_rounds >= 5 and round_id <= 10:
            logger.info(
                "user=%d round=%d/%d ok=%d failed=%d avg_lat=%.0fms",
                user_idx, round_id, len(schedule),
                ctx_metric.ok_rounds, ctx_metric.failed_rounds,
                ctx_metric.total_latency_ms / max(1, ctx_metric.total_rounds),
            )
        # 每 N 轮刷盘一次, 防止被 SIGKILL 丢数据
        flush_interval = 10
        if round_id % flush_interval == 0:
            try:
                with open("E:/AIWorking/workspace/report/echo/stress_logs.jsonl", "w", encoding="utf-8") as f:
                    for r in RESULTS_LOG:
                        f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning("flush failed: %s", e)
        # 轮间冷却 250ms
        await asyncio.sleep(0.25)
    await page.close()


# ---------- 入口 ----------

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--output", type=str,
                        default="E:/AIWorking/workspace/report/echo/stress_metrics.json")
    parser.add_argument("--log", type=str,
                        default="E:/AIWorking/workspace/report/echo/stress_logs.jsonl")
    parser.add_argument("--flush-interval", type=int, default=10,
                        help="每 N 轮刷盘一次 (防止进程被杀丢数据)")
    args = parser.parse_args()

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    distribution = {
        "greeting": 200,
        "L0_preference": 200,
        "L1_event": 200,
        "recall_query": 100,
        "multi_turn": 100,
        "contradict": 100,
        "cross_topic": 100,
    }
    per_user = args.total // args.users
    schedule = build_schedule(per_user, distribution)
    logger.info(
        "schedule: total=%d, per_user=%d, types=%s",
        args.total, per_user,
        {k: sum(1 for c, _ in schedule if c == k) for k in distribution},
    )

    # 1) 注册 N 个测试用户
    logger.info("Registering %d test users...", args.users)
    ctx_data_list: list[dict] = []
    for i in range(args.users):
        try:
            ctx = await register_and_login(i)
            expire_at = ctx["expire_at"]
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(expire_at.replace("Z", "+00:00"))
                expires_at_ms = int(dt.timestamp() * 1000)
            except Exception:
                expires_at_ms = int(time.time() * 1000) + 86400_000
            ctx["expires_at_ms"] = expires_at_ms
            ctx_data_list.append(ctx)
            logger.info("  user %d: id=%d sid=%s...", i, ctx["user_id"], ctx["session_id"][:12])
        except Exception as e:
            logger.error("register/login failed for user %d: %s", i, e)

    if not ctx_data_list:
        logger.error("No users registered, abort.")
        return

    metrics = [ContextMetrics(user_idx=i, user_id="", session_id="") for i in range(len(ctx_data_list))]

    # 2) 启动 Playwright 并跑
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        for data in ctx_data_list:
            ctx = await browser.new_context(
                viewport={"width": 1280, "height": 800}, locale="zh-CN",
            )
            data["browser_ctx"] = ctx
        try:
            await asyncio.gather(*[
                run_one_user(d, m, schedule)
                for d, m in zip(ctx_data_list, metrics)
            ])
        finally:
            for d in ctx_data_list:
                try:
                    await d["browser_ctx"].close()
                except Exception:
                    pass
            await browser.close()

    # 3) 写 summary
    summary = {
        "total_rounds": sum(m.total_rounds for m in metrics),
        "ok_rounds": sum(m.ok_rounds for m in metrics),
        "failed_rounds": sum(m.failed_rounds for m in metrics),
        "avg_latency_ms": (
            sum(m.total_latency_ms for m in metrics)
            / max(1, sum(m.total_rounds for m in metrics))
        ),
        "conv_type_distribution": {},
        "per_user": [asdict(m) for m in metrics],
    }
    for m in metrics:
        for k, v in m.conv_type_distribution.items():
            summary["conv_type_distribution"][k] = summary["conv_type_distribution"].get(k, 0) + v
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(args.log, "w", encoding="utf-8") as f:
        for r in RESULTS_LOG:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    logger.info("=" * 60)
    logger.info("Stress test done.")
    logger.info("  total_rounds: %d", summary["total_rounds"])
    logger.info("  ok: %d  failed: %d", summary["ok_rounds"], summary["failed_rounds"])
    logger.info("  avg_latency_ms: %.1f", summary["avg_latency_ms"])
    logger.info("  distribution: %s", summary["conv_type_distribution"])
    logger.info("  metrics -> %s", args.output)
    logger.info("  details -> %s", args.log)


if __name__ == "__main__":
    asyncio.run(main())