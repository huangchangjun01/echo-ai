# Echo-AI

基于多模态 Embedding + Weaviate 向量库 + 分层记忆的对话 Agent 服务。文本、图像、音频、视频统一向量化入库；对话时按意图分流，仅在检索/图片类请求下触发 RAG 跨模态检索。

---

## 核心能力

- **多模态检索**：Chinese-CLIP（图/文同空间，512 维）+ BGE-M3（文本 768 维）+ Whisper（语音转写）+ Video-MAE
- **对话记忆**：L0（长期事实）/ L1（近期摘要）/ L2，因果链跨条关联，persona 与情绪日志持久化
- **意图识别闸门**：每次 chat 先用小模型把消息分到 `chat / recall / text_search / image_search / doc_search`，按意图决定是否触发 RAG
- **ReAct + 级联**：LLM 自主调工具（search_memory / understand_image / understand_audio / analyze_emotion），首字级联（大模型续写小模型前缀）
- **流式 SSE**：实时返回 prefix / delta / resource / tool / context / done 事件，前端按帧渲染

---

## 目录结构

```
echo-ai/
├── app/
│   └── agent_runner.py            # FastAPI 入口 + lifespan（DB pool / schema / 模型预热）
├── biz/
│   ├── chat.py                    # 流式 chat：意图识别 → ReAct → 级联输出
│   └── ingest.py                  # 入库：下载 → MIME 探测 → 分块 → embedding → Weaviate
├── config/
│   ├── config.py                  # pydantic-settings 配置中心
│   └── prompts.py                 # 提示词模板（系统提示 / 抽取 / 去重 / 情感 / 意图）
├── database/
│   ├── mysql.py                   # aiomysql 连接池（同步回退到 pymysql）
│   └── schema_sql.py              # MySQL DDL + ALTER 补齐（启动时幂等）
├── embedding/
│   ├── models.py                  # 统一入口：按 modality 选模型
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
│   └── extractor.py               # 对话抽取 → 记忆落库 + 因果链
├── tools/
│   ├── search_memory.py           # 并行检索 EchoMemory + EchoDoc
│   ├── understand_image.py        # 图像描述（CLIP + LLM）
│   ├── understand_audio.py        # 语音转写 + 描述
│   └── analyze_emotion.py         # 情感分析（LLM + 关键词 fallback）
├── utils/
│   ├── downloader.py              # 异步下载 + SSRF 防护 + 大小限制
│   ├── request_context.py         # ContextVar + 日志 merge_extra
│   └── tools.py                   # LangChain BaseTool 封装
├── vector/
│   ├── vector_store.py            # EchoDoc（图/文档库）
│   └── memory_store.py            # EchoMemory（对话记忆库）
├── tests/                         # pytest + pytest-asyncio
├── run.py                         # uvicorn 启动入口
├── pyproject.toml                 # ruff + pytest 配置
├── requirements.txt
└── .env / .env.example            # 配置
```

---

## 数据存储

### MySQL（5 张表，启动时自动建表 + 补列）

| 表 | 用途 |
| --- | --- |
| `personas` | 用户人格画像 |
| `memories` | 分层记忆（`level` = L0 长期 / L1 近期摘要 / L2） |
| `memory_relations` | 记忆间的因果链（source / target / relation / weight） |
| `memory_extract_logs` | 抽取任务日志（pending / ok / error） |
| `emotion_logs` | 情感分析日志 |

### Weaviate（2 个 collection）

- **EchoDoc**：CLIP 512 维，存用户上传的图片、文本片段、音频转写（前端 RAG 资料库）
- **EchoMemory**：BGE-M3 768 维，存对话记忆、L1 摘要

---

## 意图识别闸门

每次对话入口先调小模型把消息分类，按意图分流：

| intent | multimodal_search 预注入 | L1 hint 注入 | search_memory 工具 |
| --- | :---: | :---: | :---: |
| `chat` | ❌ | ❌ | ✅ |
| `recall` | ❌ | ✅ | ✅ |
| `text_search` | ❌ | ✅ | ✅ |
| `image_search` | ✅ | ✅ | ✅ |
| `doc_search` | ❌ | ✅ | ✅ |

任何失败（超时 / 解析错误 / 标签无效）一律回退到 `chat`（最保守，不查 RAG），由 SSE `context` 事件的 `intent_source` 字段记录真实分类来源。

配置项：`MEMORY_INTENT_CLASSIFIER_ENABLED`（默认 True）、`MEMORY_INTENT_TIMEOUT_MS`（默认 1500）。

---

## 快速开始

```bash
# 1. 安装依赖
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 准备配置
cp .env.example .env   # 填入 MySQL / Weaviate / LLM / QINIU 等

# 3. 启动 Weaviate（v4，独立部署）
docker run -d --name weaviate -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  semitechnologies/weaviate:1.24.x

# 4. 启动服务
python run.py    # 监听 0.0.0.0:8000

# 5. 测试
pytest
```

---

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/chat` | 流式 SSE：意图识别 → ReAct → 级联生成（events: `context` / `tool` / `prefix` / `delta` / `resource` / `done`） |
| POST | `/ingest_file` | 入库任务，后台异步执行 |
| GET  | `/health` | 健康检查 |

SSE `context` 事件新增字段（前端可消费）：`intent` / `intent_source` / `intent_ms`。
SSE `resource` 事件字段：`url` / `name` / `modality` / `mime_type` / `similarity` —— 前端按 `modality` 决定渲染 `<img>` / `<audio>` / `<video>` / 下载链接。
