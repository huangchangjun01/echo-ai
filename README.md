# Echo-AI

一个基于 **Chinese-CLIP + Weaviate v4 + LangChain** 的多模态 RAG（检索增强生成）服务，提供文本/图像的向量化、向量检索与文件入库能力。

> 项目源码与提交记录中保留的旧说明仅供迁移参考；当前实现已切换到 **Weaviate v4** 客户端，使用 LangChain 风格的 Embeddings / Tool 抽象，并使用 `pydantic-settings` 做配置中心。

---

## 1. 业务功能

| 功能 | 说明 |
| --- | --- |
| 多模态 Embedding | 基于 `OFA-Sys/chinese-clip-vit-base-patch16`，对中文文本与图像生成同一向量空间下的表征（支持批量推理、CUDA / MPS / CPU 自动选择） |
| 文本/图像入库 | 通过七牛云对象存储下载文件，使用 libmagic 探测 MIME 后分流到文本（按 chunk_size 切片）或图像 embedding，写入 Weaviate |
| 相似度检索 | 余弦相似度检索，按阈值过滤后返回 top-k，附带真实 Weaviate UUID 与相似度得分 |
| 缓存 | 检索结果按 `(query, k, where)` 缓存到进程内 LRU+TTL，避免重复 embedding |
| Agent 工具 | `utils/tools.py` 中封装了 `VectorSearchTool`（`langchain_core.tools.BaseTool`），可被 ReAct / function-calling agent 直接调用 |
| 健康检查 | `/health` 端点，便于存活/就绪探针 |
| 入库防护 | 下载大小限制 + SSRF 防护（拒绝内网/loopback）+ tenacity 重试 |

---

## 2. 技术栈

- **Web 框架**：FastAPI 0.104 + Uvicorn 0.24（lifespan 内预热 embedding & 初始化向量库）
- **多模态模型**：Chinese-CLIP（`OFA-Sys/chinese-clip-vit-base-patch16`，HuggingFace Transformers + PyTorch）
- **向量库**：Weaviate v4 client（`weaviate-client>=4.5`，client-side embedding）
- **LLM 接入**（配置项）：Qwen（DashScope）、SiliconFlow 等 OpenAI 兼容接口
- **对象存储**：七牛云（通过 `QINIU_BASE_URL` + `fileKey` 拼装下载链接）
- **工具链**：LangChain、langchain-text-splitters、sentence-transformers、pydantic-settings、httpx、tenacity、python-magic
- **测试**：pytest + pytest-asyncio

---

## 3. 目录结构

```
echo-ai/
├── app/
│   └── agent_runner.py        # FastAPI 入口，lifespan 内预热 embedding & 初始化向量库
├── biz/
│   └── ingest.py              # 异步入库任务：下载 → MIME 探测 → 分块 → embedding → 写向量库
├── config/
│   └── config.py              # pydantic-settings 配置中心
├── embedding/
│   ├── embeddings.py          # LangChain Embeddings 实现（ChineseCLIPEmbeddings）
│   └── models.py              # Chinese-CLIP 模型加载、批量推理、设备选择、warmup
├── llm/                       # LLM 接入预留目录（待扩展）
├── utils/
│   ├── downloader.py          # 异步下载 + 大小限制 + SSRF 防护
│   └── tools.py               # LangChain BaseTool 封装（VectorSearchTool）
├── vector/
│   └── vector_store.py        # Weaviate v4 向量库封装（含 TTL 缓存、top-k 检索）
├── tests/                     # 单元测试（mock embedder / mock weaviate / TestClient）
├── .github/workflows/ci.yml   # GitHub Actions：lint + test
├── pyproject.toml             # ruff + pytest 配置
├── run.py                     # uvicorn 启动入口
├── requirements.txt
├── .env                       # 运行配置（含密钥，请勿提交到公开仓库）
└── .env.example               # 配置示例
```

---

## 4. 环境与配置

`.env` 中的关键项（完整列表见 `.env.example`）：

```ini
# Weaviate
WEAVIATE_URL=                       # 优先使用；为空时按 host/scheme/port 拼接
WEAVIATE_HOST=localhost
WEAVIATE_SCHEME=http
WEAVIATE_PORT=8080
WEAVIATE_CLASS=EchoDoc
WEAVIATE_API_KEY=

# 七牛云对象存储
QINIU_BASE_URL=tfpdkiq9g.hn-bkt.clouddn.com
QINIU_ALLOWED_SUBDOMAINS=           # 留空时自动从 BASE_URL 推断

# Embedding
EMBEDDING_MODEL_NAME=OFA-Sys/chinese-clip-vit-base-patch16
EMBEDDING_DEVICE=auto               # auto | cuda | mps | cpu
EMBEDDING_DIM=512
EMBEDDING_BATCH_SIZE=8
EMBEDDING_CHUNK_SIZE=256
EMBEDDING_CHUNK_OVERLAP=32
EMBEDDING_WARMUP_ON_START=true

# Ingest
INGEST_MAX_DOWNLOAD_BYTES=52428800  # 50 MB
INGEST_DOWNLOAD_TIMEOUT_SECONDS=60
INGEST_DOWNLOAD_RETRIES=3
INGEST_ENABLE_CHUNKING=true
INGEST_CACHE_TTL_SECONDS=60
INGEST_CACHE_MAXSIZE=256

# 检索阈值（余弦相似度，范围 [-1, 1]，默认 0.7）
VECTOR_SIMILARITY_THRESHOLD=0.7
```

> 通过 `1 - distance` 计算余弦相似度，命中阈值后才会进入返回列表；top-k 由客户端传入。

---

## 5. API 速查

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/chat` | 检索 + 返回 top-k 候选（已校验 query/k） |
| POST | `/search` | `/chat` 的简化版别名 |
| POST | `/ingest_file` | 入库任务，后台异步执行 |
| GET  | `/health` | 存活/就绪探针，返回 `{"status":"ok"}` |

---

## 6. 快速开始

```powershell
# 1. 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 3. 准备 .env（参考 .env.example）

# 4. 启动服务
python run.py
# 监听 0.0.0.0:8000；run.py 自动检测 PyCharm Debug 模式

# 5. 运行测试
pytest
```

> Weaviate 服务本身需单独启动（v4）。可通过 Docker：
> `docker run -d --name weaviate -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true semitechnologies/weaviate:1.24.x`

---

## 7. 已知未实现

- **第 1 点**（`/chat` 接入 LLM 生成最终答案）—— 保留 `/chat` 仅返回候选，由上层服务拼 prompt。
- **第 13 点**（Prometheus 指标 / 探活式 health）—— `/health` 仅返回 `{"status":"ok"}`，未暴露 `/metrics`，监控按需再加。
- **第 17 点**（API Key / JWT / CORS）—— 当前端点无鉴权，对外暴露前需加 `CORSMiddleware` + FastAPI `Security`。