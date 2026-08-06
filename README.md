# Echo-AI

基于多模态 Embedding + Weaviate 向量库 + **两层记忆**（对话记忆 / 回忆记忆）的 Agent 服务。
文本、图像、音频、视频统一向量化入库；对话时按意图分流，仅在检索/回忆/图片类请求下触发 RAG 跨模态检索与「回忆记忆」摘要回灌。

> 本版本（v2.x）的核心扩展：在原有「对话记忆 (EchoMemory)」之外，新增独立的「**回忆记忆 (EchoRecall)**」维度——
> 通过异步解析用户上传的多模态源文件 → 生成结构化 `.md` → 直传对象存储 → 写入 EchoRecall 摘要向量，
> 支持"按主题回顾/追问细节"等长生命周期场景。回忆记忆与对话记忆完全隔离，**永不遗忘**。

---

## 核心能力

- **多模态检索**：Chinese-CLIP（图/文同空间，512 维）+ BGE-M3（文本 768/1024 维）+ Whisper（语音转写）+ Video-MAE
- **两层记忆**：
  - **对话记忆**：`personas` + `memories`（L0 长期事实 / L1 近期摘要 / L2）+ `memory_relations` 因果链；带**遗忘机制**（retain_score 衰减 + GC 循环）
  - **回忆记忆 (EchoRecall)**：独立的 `recall_memory` 表 + `EchoRecall` 向量集合 + 对象存储 `.md`；**永不遗忘**，按 `memoryId` 幂等 upsert
- **意图识别闸门**：每次 chat 先用小模型把消息分到 `chat / recall / text_search / image_search / doc_search`，按意图决定是否触发 RAG
- **ReAct + 级联**：LLM 自主调工具（`search_memory` / `read_memory_full` / `understand_image` / `understand_audio` / `analyze_emotion`），首字级联（大模型续写小模型前缀）
- **流式 SSE**：实时返回 `context` / `prefix` / `delta` / `resource` / `tool` / `done` 事件，前端按帧渲染
- **多模态解析器 (parsers)**：文本/图片/音频/视频独立解析；视频走「场景分段 + 结构化聚合（人物/动作/物品/…）」；按扩展名兜底（避免 `mp4` 被前端误标成文本）

---

## 本轮变更摘要（2026-08）

围绕「回忆记忆」做了完整的工程化重构，新增 7 个模块、4 个对外路由、2 张新表、1 个向量集合：

| 类别 | 变更 |
| --- | --- |
| 新模块 | `parsers/`（text/image/audio/video + scene_aggregator）、`skills/`（`build_memory_md` 等 LLM 技能）、`storage/qiniu_client.py`（直传/删除）、`biz/recall.py`、`biz/recall_search.py`、`biz/chat_memory.py`、`utils/temp_files.py`、`vector/recall_store.py`、`database/recall_status.py`、`tools/read_memory_full.py` |
| 新路由 | `POST /memory/parse`、`POST /memory/reparse`、`POST /memory/source/delete`、`POST /memory/delete`（统一在 `app/routers/recall.py`） |
| 新表 | `chat_sessions`（对话会话摘要 + 遗忘权重）、`chat_messages`（短期多轮消息缓冲） |
| 新集合 | `EchoRecall`（BGE-M3 768/1024 维，独立 class、阈值） |
| 新工具 | `read_memory_full(memory_id, user_id)`：把命中摘要展开为整段 `.md`，由 LLM 在 ReAct 中按需调用 |
| 配置 | `WEAVIATE_RECALL_CLASS`、`WEAVIATE_RECALL_THRESHOLD`、`EMBEDDING_ENDPOINT` / `BGE_M3_ENDPOINT`（HF 镜像）、`QINIU_ACCESS_KEY/SECRET_KEY/BUCKET_NAME`；`MEMORY_INTENT_TIMEOUT_MS` 提升至 40s（适配推理型小模型） |
| Chat 改造 | `/chat` 在 ReAct 之前先做一次 EchoRecall 摘要检索；将命中 `memoryId` 注入 ReAct 上下文；prompt 显式告诉 LLM「摘要不够时可调 `read_memory_full` 拿全文」；`user_id` 一律用 session 真值强制覆盖 LLM 提交值 |

详细说明见各模块注释；本文档下文按主题逐一展开。

---

## 回忆记忆（EchoRecall）

与对话记忆并列的另一条存储链路，专注于「**用户主动沉淀的生活/工作记忆**」。

### 设计原则

1. **md 是记忆的唯一完整载体**：EchoRecall 只索引摘要；细节全部落在对象存储的 `.md` 里（`memory/{userId}/{roleId}/{memoryId}/{memoryId}.md`）。
2. **隔离维度**：`userId × roleId × memoryId`；编辑后 `upsert`（先删后插）。
3. **永不遗忘**：与对话记忆的 `retain_score` 衰减不同，回忆记忆一旦写入即长期保留，仅支持显式 `delete_memory` / `delete_source`。
4. **异步解析**：`/memory/parse` 立即返回，后台跑「抢锁 → 多模态解析 → 生成 md → 直传对象 → 同步缓存到 echo-core → 写 EchoRecall → 解锁」。
5. **渐进式回忆**：搜索只回摘要；LLM 在 ReAct 中调 `read_memory_full` 才拉整段 `.md`，避免无差别放大上下文。

### 解析链路

```
echo-core 事务后
   └── POST /memory/parse  ──► parse_memory()
                                  ├── try_acquire_edit_lock (CAS, 60s)
                                  ├── 并发解析所有源文件 (Semaphore=4)
                                  │     ├── parse_text   (1)
                                  │     ├── parse_image  (2, CLIP+LLM)
                                  │     ├── parse_video  (3, 帧采样+场景聚合)
                                  │     └── parse_audio  (4, Whisper 转写)
                                  ├── skills.build_memory_md()  生成 5 大节 md
                                  ├── qiniu.upload_bytes(md_key, …)  直传对象
                                  ├── echo-core /api/memory/md-content  同步缓存
                                  ├── RecallVectorStore.upsert(memoryId, summary, …)
                                  └── release_edit_lock
```

### 视频场景结构化

视频不再产出长 prose 描述，而是 LLM Stage 1 输出 8 字段结构化 JSON（人物/场景/动作/物品/文字/表情/颜色方位/变化）+ 1~2 句连贯 prose；`scene_aggregator` Stage 2 做「贯穿主体 + 精简 per-scene」聚合：

- 人物/物品/文字：union 跨场景，只在「贯穿主体」出现一次
- 场景：取众数（出现 ≥ ⌈N/2⌉ 次）
- 动作/表情：仅 per-scene
- 变化：永远保留
- Stage 3 由 `skills` 用 prose 直接渲染最终片段

### 文件类型冲突兜底

`parsers/registry.py` 维护 `_EXT_TO_FILE_TYPE` 映射，`fileType` 与扩展名冲突时**扩展名权威**——避免 mp4 字节流被前端误标为 1（文本）后 decode 成 ftyp/moov 元数据污染 prompt。

---

## 目录结构

```
echo-ai/
├── app/
│   ├── agent_runner.py            # FastAPI 入口 + lifespan（DB pool / schema / 模型预热 / GC 循环）
│   └── routers/
│       └── recall.py              # /memory/parse、/memory/reparse、/memory/source/delete、/memory/delete
├── biz/
│   ├── chat.py                    # 流式 chat：意图识别 → EchoRecall 预检索 → ReAct → 级联输出
│   ├── chat_memory.py             # 对话短期记忆：chat_sessions + chat_messages + retain_score GC
│   ├── recall.py                  # 回忆记忆编排：parse_memory / reparse_memory / delete_source / delete_memory
│   ├── recall_search.py           # EchoRecall top5 检索 + md 拉取（经 echo-core 代理）
│   └── ingest.py                  # 入库：下载 → MIME 探测 → 分块 → embedding → Weaviate
├── config/
│   ├── config.py                  # pydantic-settings 配置中心（含 recall_class/threshold、HF 镜像）
│   └── prompts.py                 # 提示词模板（系统提示 / 抽取 / 去重 / 情感 / 意图）
├── database/
│   ├── mysql.py                   # aiomysql 连接池（同步回退到 pymysql）
│   ├── schema_sql.py              # MySQL DDL + ALTER 补齐（含 chat_sessions / chat_messages）
│   └── recall_status.py           # recall_memory 表：parse_status / edit_status CAS 锁
├── embedding/
│   ├── models.py                  # 统一入口：按 modality 选模型（HF 镜像环境注入）
│   ├── clip.py                    # Chinese-CLIP（图文跨模态）
│   ├── bge_m3.py                  # BGE-M3（文本）
│   ├── whisper.py                 # 语音转写
│   └── video_mae.py               # 视频
├── llm/
│   ├── client.py                  # OpenAI 兼容客户端（大模型 + 小模型）
│   ├── cascade.py                 # 大小模型级联（首字延迟优化）
│   ├── react.py                   # ReAct 循环（决策 → 派发工具 → 续写）
│   └── intent.py                  # 意图分类（小模型 JSON 输出 + fallback）
├── memory/
│   ├── retriever.py               # L0/L1 加载 + 多模态跨模态检索 + persona
│   ├── extractor.py               # 对话抽取 → 记忆落库 + 因果链
│   ├── archiver.py                # 长期归档
│   ├── extract.py                 # 抽取辅助
│   └── retrieve.py                # 检索辅助
├── parsers/                       # 多模态解析器（pluggable，按 fileType/扩展名分派）
│   ├── base.py                    # ParsedFile / ParsedChunk 数据结构
│   ├── registry.py                # 扩展名权威 + fileType 分派
│   ├── text_parser.py             # 文本 / 文档
│   ├── image_parser.py            # CLIP + LLM 描述
│   ├── audio_parser.py            # Whisper 转写 + 描述
│   ├── video_parser.py            # 帧采样 + 场景分段 + Stage 1 结构化
│   └── scene_aggregator.py        # Stage 2：贯穿主体 + 精简 per-scene
├── skills/                        # LLM Skill：结构化生成 / 校验 / 补齐
│   ├── __init__.py                # build_memory_md / parse_summary / update_source_metadata / remove_source_section
│   └── create_memory_md/SKILL.md  # Skill 文档
├── storage/
│   └── qiniu_client.py            # 七牛直传/删除（带 AK/SK 单例）
├── tools/
│   ├── search_memory.py           # 并行检索 EchoMemory + EchoDoc + EchoRecall
│   ├── read_memory_full.py        # 把某 memoryId 的整段 .md 拉回 LLM 上下文
│   ├── understand_image.py        # 图像描述（CLIP + LLM）
│   ├── understand_audio.py        # 语音转写 + 描述
│   └── analyze_emotion.py         # 情感分析（LLM + 关键词 fallback）
├── utils/
│   ├── downloader.py              # 异步下载 + SSRF 防护 + 大小限制
│   ├── request_context.py         # ContextVar + 日志 merge_extra
│   ├── temp_files.py              # 临时工作目录（解析器下载/处理自动清理）
│   └── tools.py                   # LangChain BaseTool 封装
├── vector/
│   ├── vector_store.py            # EchoDoc（图/文档库）
│   ├── memory_store.py            # EchoMemory（对话记忆库）
│   └── recall_store.py            # EchoRecall（回忆记忆库）
├── tests/                         # pytest + pytest-asyncio
├── run.py                         # uvicorn 启动入口
├── pyproject.toml                 # ruff + pytest 配置
├── requirements.txt               # 含 qiniu SDK
└── .env / .env.example            # 配置
```

---

## 数据存储

### MySQL（7 张表，启动时自动建表 + 补列）

| 表 | 用途 |
| --- | --- |
| `personas` | 用户人格画像 |
| `memories` | 对话分层记忆（`level` = L0 长期 / L1 近期摘要 / L2） |
| `memory_relations` | 记忆间的因果链（source / target / relation / weight） |
| `memory_extract_logs` | 抽取任务日志（pending / ok / error） |
| `emotion_logs` | 情感分析日志 |
| `chat_sessions` | 对话会话摘要 + `retain_score` 遗忘权重 + 归档标记 |
| `chat_messages` | 短期多轮消息缓冲（按 session_id 倒序取最近 N 条） |

> 回忆记忆表 `recall_memory` 由 echo-core 维护；echo-ai 仅通过 `database/recall_status.py` 做 `parse_status` / `edit_status` 的轻量更新与 CAS 锁。

### Weaviate（3 个 collection）

- **EchoDoc**：Chinese-CLIP 512 维，存用户上传的图片、文本片段、音频转写（前端 RAG 资料库）
- **EchoMemory**：BGE-M3 768/1024 维，存对话记忆、L1 摘要
- **EchoRecall**：BGE-M3 768/1024 维，存回忆记忆的**摘要**（细节落在对象存储的 `.md`）

阈值按集合分别配置（共用阈值会过滤掉 CLIP 命中）：

| 集合 | 字段 | 默认 | 含义 |
| --- | --- | --- | --- |
| EchoDoc | `WEAVIATE_DOC_THRESHOLD` | 0.25 | CLIP 跨模态 cosine |
| EchoMemory | `WEAVIATE_MEMORY_THRESHOLD` | 0.7 | BGE-M3 文本相似 |
| EchoRecall | `WEAVIATE_RECALL_THRESHOLD` | 0.5 | 摘要召回可适当放宽 |

---

## 意图识别闸门

每次对话入口先调小模型把消息分类，按意图分流：

| intent | multimodal_search 预注入 | L1 hint 注入 | search_memory 工具 | read_memory_full 提示 |
| --- | :---: | :---: | :---: | :---: |
| `chat` | ❌ | ❌ | ✅ | ❌ |
| `recall` | ❌ | ✅ | ✅ | ✅（强烈） |
| `text_search` | ❌ | ✅ | ✅ | ✅ |
| `image_search` | ✅ | ✅ | ✅ | ❌ |
| `doc_search` | ❌ | ✅ | ✅ | ✅ |

任何失败（超时 / 解析错误 / 标签无效）一律回退到 `chat`（最保守，不查 RAG），由 SSE `context` 事件的 `intent_source` 字段记录真实分类来源。

配置项：`MEMORY_INTENT_CLASSIFIER_ENABLED`（默认 True）、`MEMORY_INTENT_TIMEOUT_MS`（默认 **40000**，适配推理型小模型；非推理模型可调回 1500）。

---

## Chat 链路（v2.x）

```
POST /chat {userId, roleId, sessionId?, message, stream?}
  │
  ├── 1) 意图识别（小模型 JSON + fallback 到 chat）
  ├── 2) 加载 persona + L0/L1 对话记忆
  ├── 3) ★ 新增：EchoRecall top5 摘要检索（userId × roleId）
  │        └─ 命中 memoryId 注入 ReAct 决策上下文
  ├── 4) ReAct 决策循环（react_max_iter）
  │     ├─ search_memory (EchoMemory + EchoDoc + EchoRecall)
  │     ├─ read_memory_full (按 memoryId 拉整段 .md)
  │     ├─ understand_image / understand_audio
  │     └─ analyze_emotion
  │     工具入参 user_id/session_id/role_id 一律用 session 真值强制覆盖
  ├── 5) cascade 输出（小模型前缀 → 大模型续写 → SSE delta）
  └── 6) fire-and-forget：extract_and_archive_async（对话记忆抽取）
```

---

## 快速开始

```bash
# 1. 安装依赖
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 准备配置
cp .env.example .env   # 填入 MySQL / Weaviate / LLM / QINIU_AK / QINIU_SK / QINIU_BUCKET

# 3. 启动 Weaviate（v4，独立部署）
docker run -d --name weaviate -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  semitechnologies/weaviate:1.24.x

# 4. 启动服务
python run.py    # 监听 0.0.0.0:8000

# 5. 测试
pytest
```

> 国内部署务必设置 `EMBEDDING_ENDPOINT=https://hf-mirror.com` 与 `BGE_M3_ENDPOINT=https://hf-mirror.com`，否则首次加载模型会在 SSL 阶段失败。

---

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET  | `/health` | 健康检查 |
| POST | `/chat` | 流式 SSE：意图识别 → EchoRecall 预检索 → ReAct → 级联生成（events: `context` / `tool` / `prefix` / `delta` / `resource` / `done`） |
| POST | `/ingest_file` | 入库任务，后台异步执行 |
| POST | `/memory/parse` | 解析回忆记忆：异步抢锁 → 多模态解析 → 生成 md → 直传对象 → 写 EchoRecall |
| POST | `/memory/reparse` | 编辑后再解析：先清旧向量 → 走标准解析链路（stage=`recall_reparse`） |
| POST | `/memory/source/delete` | 删除某文件对应片段 → 重算摘要 → 覆盖原 md → 更新向量 |
| POST | `/memory/delete` | 删除整条 EchoRecall 向量记录（对象存储与 MySQL 由 echo-core 处理） |

SSE `context` 事件新增字段（前端可消费）：`intent` / `intent_source` / `intent_ms`。
SSE `resource` 事件字段：`url` / `name` / `modality` / `mime_type` / `similarity` —— 前端按 `modality` 决定渲染 `<img>` / `<audio>` / `<video>` / 下载链接。

---

## 安全 / 资源约束

- **SSRF 防护**：`utils/downloader.py` 校验白名单 + 字节上限 + 重试
- **鉴权**：`read_memory_full` 强制校验 `memory_id` 形状与 `user_id` 归属；工具入参 `user_id` 一律覆盖 LLM 提交值
- **HF 镜像**：通过 `HF_ENDPOINT` 进程级注入，避免与本机 `huggingface_hub` 默认端点冲突
- **edit_lock CAS**：`try_acquire_edit_lock` 60s 轮询上限，避免并发写入 EchoRecall
- **临时文件**：解析器统一通过 `temp_workspace()` 拿目录，with 退出自动清理