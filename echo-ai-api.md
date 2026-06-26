# Echo-AI Service API Skill

## 概述

Echo-AI 是一个基于 FastAPI 的 AI Agent 服务，提供语义搜索、文件摄入、OpenAI 兼容对话补全、记忆管理等核心能力。服务运行在 `http://0.0.0.0:8000`。

---

## 一、HTTP API 接口

### 1. GET /health

**用途**: 健康检查，供 uptime 探针调用。

**请求参数**: 无

**响应示例**:
```json
{
  "status": "ok"
}
```

---

### 2. POST /chat

**用途**: 基于向量检索的语义搜索对话。根据用户查询在 Weaviate 向量库中检索相关文档，返回 top-k 候选项。

**请求体** (JSON):
```json
{
  "query": "string (1-2048 字符, 必填)",
  "k": 5 (1-50, 默认 5),
  "userId": "string (1-128 字符, 必填)"
}
```

**响应示例**:
```json
{
  "query": "原始查询",
  "k": 5,
  "userId": "user123",
  "candidates": [
    {
      "id": "文档ID",
      "document": "文档内容",
      "metadata": { "fileId": "...", "fileName": "...", "userId": "...", "chunkIndex": 0 },
      "distance": 0.05
    }
  ]
}
```

**错误码**:
- `400`: 无效查询（如空向量）
- `500`: 服务器内部错误

---

### 3. POST /search

**用途**: 与 `/chat` 接口等价，语义搜索别名。

**请求体** (JSON):
```json
{
  "query": "string (1+ 字符, 必填)",
  "k": 5 (1-50, 默认 5),
  "userId": "string (1-128 字符, 必填)"
}
```

**响应**: 同 `/chat`

---

### 4. POST /ingest_file

**用途**: 异步摄入文件到向量库。支持文本和图片文件，后台自动下载、分块、向量化、存储。

**请求体** (JSON):
```json
{
  "userId": "string (必填)",
  "file": {
    "fileId": "string (必填)",
    "fileName": "string (必填)",
    "fileKey": "string (可选, 七牛云文件key)",
    "url": "string (可选, 文件直链)"
  }
}
```

**URL 解析优先级**: `url` > `fileKey` (拼接七牛云 base_url)

**支持的文件类型**:
- 图片: `image/png`, `image/jpeg`, `image/jpg`, `image/webp`, `image/gif`, `image/bmp`
- 文本: `text/plain`, `text/markdown`, `text/csv`, `text/x-python`, `application/json`, `application/x-ndjson`, `text/x-log`
- 不支持: `video/mp4`, `video/webm`, `audio/mpeg`, `audio/ogg`, `audio/flac`, `application/zip`, `application/octet-stream`

**响应示例**:
```json
{
  "ok": true,
  "queued": true,
  "fileId": "file_001"
}
```

---

### 5. POST /v1/chat/completions

**用途**: OpenAI 兼容的对话补全接口，支持 ReAct 循环（推理→工具调用→观察→推理）和 SSE 流式输出。自动注入 L0 核心记忆作为系统提示词。

**前置依赖**:
- 大模型配置 (`LARGE_LLM_*`) 必须可用
- 数据库 (`DB_*`) 必须可用

**请求体** (JSON):
```json
{
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "userId": "string (1-128 字符, 必填)",
  "stream": false,
  "k": 5 (1-50, 默认 5)
}
```

**非流式响应** (`stream: false`):
```json
{
  "id": "chatcmpl-xxxx",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "回复内容"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

**流式响应** (`stream: true`):
```
data: {"token": "你好"}
data: {"token": "！"}
data: [DONE]
```

**ReAct 循环行为**:
1. 构建系统提示词（含人格设定 + L0 核心记忆）
2. 调用大模型（带 tools 定义）
3. 若有 tool_calls → 执行工具 → 反馈结果 → 回到步骤 2
4. 若无 tool_calls → 返回最终回复（流式输出最终回复的 token）
5. 最多循环 5 次
6. 请求完成后异步触发记忆抽取和归档

**错误码**:
- `503`: LLM 或 Memory 服务不可用
- `500`: LLM 推理失败

---

### 6. POST /v1/memory/status

**用途**: 查询用户各层记忆数量及最新摘要（调试用）。

**前置依赖**: MySQL 数据库必须可用

**请求体** (JSON):
```json
{
  "userId": "string (1-128 字符, 必填)"
}
```

**响应示例**:
```json
{
  "user_id": "user123",
  "l0_count": 5,
  "l1_count": 23,
  "l2_count": 150,
  "latest_summary": "用户偏好喝咖啡，最近在研究机器学习..."
}
```

---

## 二、内部模块接口

### LLM 模块 (`llm/`)

#### `llm.inference.call_llm()`
```python
async def call_llm(
    *,
    base_url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]
```
非流式调用 OpenAI 兼容 API，返回完整响应 dict。

#### `llm.inference.call_llm_stream()`
```python
async def call_llm_stream(
    *,
    base_url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    extra_body: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any], None]
```
流式调用 OpenAI 兼容 API，yield 每个 SSE chunk dict。

#### `llm.inference.SmallModelInference`
小模型推理类（默认 Qwen2.5-1.5B），提供 `generate()` 和 `generate_stream()` 方法。

#### `llm.inference.LargeModelInference`
大模型推理类（默认 qwen3.5-plus），提供 `generate()` 和 `generate_stream()` 方法。

#### `llm.inference.cascaded_stream()`
```python
async def cascaded_stream(
    messages: list[dict[str, Any]],
    *,
    small_model: SmallModelInference | None = None,
    large_model: LargeModelInference | None = None,
) -> AsyncGenerator[str, None]
```
级联流式推理：小模型快速生成前缀 → 大模型深度续写。

#### `llm.react.ReActLoop`
```python
class ReActLoop:
    def __init__(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        memory_context: str = "",
        max_iterations: int = 5,
        model_config: dict[str, Any] | None = None,
    ) -> None

    async def run(self) -> str
    async def run_stream(self) -> AsyncGenerator[str, None]
```
ReAct 循环：推理 → 行动 → 观察 → 推理，最多 5 轮循环。

#### `llm.react.register_tool()` / `build_tool_definitions()` / `execute_tool_call()`
工具注册、定义构建和执行路由。

---

### Memory 模块 (`memory/`)

#### `memory.retrieve.retrieve_l0_memories(user_id) -> list[dict]`
加载用户 L0 核心记忆。

#### `memory.retrieve.retrieve_l1_memories(user_id, query, k=5) -> list[dict]`
使用 Weaviate 向量检索 Top-K 相关 L1 记忆（优先 BGE-M3，回退 Chinese-CLIP）。

#### `memory.retrieve.retrieve_causal_chain(user_id, memory_id) -> list[dict]`
递归查询因果关系链（从 memory_relations 表）。

#### `memory.retrieve.retrieve_cross_modal(user_id, query, modality="text") -> list[dict]`
跨模态检索：text(默认BGE-M3) / image(CLIP) / audio(Whisper)。

#### `memory.retrieve.retrieve_combined(user_id, query, k=5) -> dict`
组合检索：L0 + L1 + 因果链。

#### `memory.extract.extract_and_archive(user_id, messages) -> dict`
综合记忆抽取流水线：抽取 → 去重 → 向量化 → 归档 → 关系识别 → 情感标注 → 合并清理 → 摘要。
返回 `{"archived_count": int, "relations_count": int, "summary": str}`。

#### `memory.extract.extract_atomic_facts(messages) -> list[dict]`
使用大模型从对话中抽取原子事实和因果关系。

#### `memory.extract.semantic_dedup(facts, user_id) -> list[dict]`
语义去重：向量化 + L1 检索 + LLM 判重。

#### `memory.extract.tag_emotion(fact) -> dict`
使用小模型对事实进行情感标注（joy/sadness/anger/fear/surprise/disgust/neutral/anticipation/trust）。

#### `memory.extract.merge_similar_memories(user_id)`
合并相似度 > merge_threshold 的记忆，使用 LLM 合并。

#### `memory.extract.resolve_contradictions(user_id)`
检测并清理矛盾记忆。

#### `memory.extract.generate_summary(user_id) -> str`
基于 L0 记忆生成摘要。

#### `memory.archiver.archive_memory(...) -> int`
归档记忆到 core_memories 表，根据 importance 判定层级 (L0/L1/L2)，超限自动淘汰。

#### `memory.archiver.archive_relation(source_id, target_id, relation_type, confidence) -> int`
写入记忆关系到 memory_relations 表。

#### `memory.archiver.archive_summary(user_id, summary, memory_ids) -> int`
写入记忆摘要到 memory_summaries 表。

#### `memory.archiver.get_l0_memories(user_id) -> list[dict]`
查询 L0 层核心记忆。

---

### Tools 模块 (`tools/`)

已注册的内置工具：

| 工具名 | 文件 | 描述 |
|--------|------|------|
| `search_memory` | `tools/search_memory.py` | 搜索用户记忆：向量检索 + L0 核心记忆 |
| `analyze_emotion` | `tools/analyze_emotion.py` | 分析文本情感（joy/sadness/anger/fear/surprise/disgust/neutral） |
| `understand_audio` | `tools/understand_audio.py` | 使用 Whisper 将音频转录为文本 |
| `understand_image` | `tools/understand_image.py` | 使用视觉模型理解图片内容 |
| `vector_search` | `utils/tools.py` | 在 Weaviate 中按语义相似度搜索用户文档（LangChain BaseTool） |

**工具注册接口**:
```python
# 注册工具
from tools import register_tool
register_tool("tool_name", tool_fn, definition_fn)

# 获取工具
from tools import get_tool, get_all_tools
tool = get_tool("tool_name")

# 构建 OpenAI 兼容的工具定义
from tools import build_tool_definitions
defs = build_tool_definitions()
```

---

### Vector Store 模块 (`vector/`)

#### `vector.vector_store.WeaviateVectorStore`
```python
class WeaviateVectorStore:
    def add_texts(self, ids, texts, metadatas=None, embeddings=None) -> list[str]
    def query(self, query_text, n_results=5, embedding_fn=None, where=None) -> dict
    def get_document(self, doc_id) -> dict | None
    def delete(self, ids) -> None
    def close(self) -> None
```
Weaviate 向量存储，通过 REST API 操作（无 gRPC 依赖）。`query()` 返回 Chroma 风格响应：
```python
{"ids": [["id1"]], "documents": [["text1"]], "metadatas": [[{"fileId": "..."}]], "distances": [[0.05]]}
```

#### `vector.vector_store.get_vector_store() -> WeaviateVectorStore`
进程级单例。

#### `vector.vector_store.reset_vector_store()`
释放单例，关闭连接。

---

### Embedding 模块 (`embedding/`)

#### `embedding.embeddings.ChineseCLIPEmbeddings`
```python
class ChineseCLIPEmbeddings(Embeddings):
    def embed_documents(self, texts) -> list[list[float]]
    def embed_query(self, text) -> list[float]
    def embed_images(self, images) -> list[list[float]]
    def embed_image(self, image) -> list[float]
    def warmup(self) -> None
```
Chinese-CLIP 多模态 Embedding，支持文本和图像向量化。回退链：CLIP → sentence-transformers → SHA256。

#### `embedding.models` 核心函数
- `compute_text_embeddings(texts, device="auto")` - 文本批量向量化
- `compute_image_embeddings(images, device="auto")` - 图像批量向量化
- `compute_text_embedding(text)` - 单文本向量化
- `compute_image_embedding(image_data)` - 单图像向量化
- `warmup(device="auto")` - 模型预热

#### `embedding.bge_m3`
- `compute_query_embedding(text)` - BGE-M3 查询向量化 (768维)
- `compute_embeddings(texts)` - BGE-M3 批量向量化

#### `embedding.whisper`
- `transcribe(audio_data)` - Whisper 音频转录
- `extract_voiceprint(audio_text)` - 提取声纹特征向量

#### `embedding.video_mae`
- VideoMAE 视频理解模型（MCG-NJU/videomae-base）

---

### Database 模块 (`database/`)

#### `database.mysql`
```python
async def init_db() -> None          # 创建所有表（幂等）
async def close_db() -> None         # 关闭引擎
async def get_session() -> AsyncIterator[AsyncSession]  # 异步会话上下文管理器
```

#### `database.models` 数据表
- **core_memories**: 核心记忆表 (`id`, `user_id`, `content`, `memory_type`, `layer`, `importance`, `emotion_tag`, `emotion_intensity`, `weaviate_uuid`, `embedding_dim`, `summary`, `created_at`, `updated_at`)
- **memory_relations**: 记忆关系表 (`id`, `source_id`, `target_id`, `relation_type`, `confidence`, `created_at`)
- **memory_summaries**: 记忆摘要表 (`id`, `user_id`, `summary`, `memory_ids`, `created_at`)

---

### Ingest 模块 (`biz/`)

#### `biz.ingest.ingest_file(user_id, file_obj, embeddings, vectorstore) -> IngestResult`
```python
@dataclass
class IngestResult:
    success: bool
    file_id: str
    chunks: int = 0
    error: str | None = None
```
下载、分类、向量化、持久化单个文件。支持文本分块（RecursiveCharacterTextSplitter）和图片多模态 Embedding。

---

### Config 模块 (`config/`)

#### `config.config.get_settings() -> Settings`
懒加载单例配置。支持 `.env` 文件和环境变量。

**配置项**:
- `WEAVIATE_*`: Weaviate 连接配置（url, host, scheme, port, class, api_key）
- `QINIU_*`: 七牛云配置（base_url, allowed_subdomains）
- `EMBEDDING_*`: Embedding 配置（model_name, device, dim, batch_size, chunk_size, chunk_overlap, warmup_on_start）
- `INGEST_*`: 摄入配置（max_download_bytes, download_timeout_seconds, download_retries, enable_chunking, cache_ttl_seconds, cache_maxsize）
- `DB_*`: MySQL 数据库配置（host, port, user, password, name）
- `SMALL_LLM_*`: 小模型配置（base_url, model, api_key, max_tokens, temperature）
- `LARGE_LLM_*`: 大模型配置（base_url, model, api_key, max_tokens, temperature）
- `MEMORY_*`: 记忆系统配置（l0_max_count, l0_min_importance, l1_max_count, l1_min_importance, l2_max_count, dedup_threshold, merge_threshold, contradiction_threshold, max_summary_length, max_summaries）
- `BGE_M3_*`: BGE-M3 多模态 Embedding 配置
- `CLIP_*`: Chinese-CLIP 配置
- `WHISPER_*`: Whisper 配置
- `VIDEO_MAE_*`: VideoMAE 配置
- `VECTOR_SIMILARITY_THRESHOLD`: 向量相似度阈值 (默认 0.7)

---

## 三、启动方式

```bash
# 普通模式
python run.py

# 或直接
uvicorn app.agent_runner:app --host 0.0.0.0 --port 8000
```

服务启动时自动执行：
1. 初始化 MySQL 连接池（超时 5s，失败不阻塞启动）
2. 注册所有工具到 ReAct 循环
3. Embedding 模型预热（可选）
4. 初始化 Weaviate 向量存储

---

## 四、架构依赖

```
用户请求
  ├── /chat, /search → WeaviateVectorStore.query() → ChineseCLIPEmbeddings
  ├── /ingest_file → ingest_file() → download → classify → embed → vector_store.add_texts()
  ├── /v1/chat/completions → ReActLoop → call_llm() → execute_tool_call() → memory.extract_and_archive()
  └── /v1/memory/status → MySQL (core_memories, memory_summaries)
```