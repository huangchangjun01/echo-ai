# 重构报告

## 第一部分：删除的部分

本次重构无删除文件，均为新增和扩展。

原有的单文件工具模块 `utils/tools.py` 中的 `VectorSearchTool` 被保留，但其功能已由新的 `tools/` 模块增强，不再作为唯一的工具实现。所有原有文件均得以保留，重构通过新增模块和扩展现有文件来完成架构升级，未对任何已有代码进行破坏性删除。

---

## 第二部分：保留的部分（修改前后对比）

### 修改文件对比

| 文件 | 修改前 | 修改后 |
|------|--------|--------|
| `config/config.py` | 仅包含 Weaviate、Qiniu、Embedding、Ingest 配置 | 新增 MySQL、LLM（小/大模型）、记忆系统、多模态 Embedding（BGE-M3、CLIP、Whisper、VideoMAE）配置 |
| `.env` | 仅包含 Weaviate、Qiniu、LLM 旧配置 | 新增 MySQL 连接、LLM 小/大模型分离配置、记忆系统参数、多模态 Embedding 配置 |
| `.env.example` | 仅包含基础配置示例 | 同步新增所有配置项示例 |
| `app/agent_runner.py` | 仅有 `/chat`（向量检索）、`/search`、`/ingest_file`、`/health` 接口 | 新增 `/v1/chat/completions`（完整对话 + ReAct + 工具调用 + 流式输出）、`/v1/memory/status`；扩展 lifespan 初始化 MySQL |
| `embedding/__init__.py` | 仅导出 `ChineseCLIPEmbeddings` | 新增导出 BGE-M3、Whisper、VideoMAE 接口 |
| `requirements.txt` | 基础依赖 | 新增 `aiomysql`、`sqlalchemy`、`openai`、`opencv-python-headless`、`imageio`、`soundfile` |

### 新增模块

| 模块 | 说明 |
|------|------|
| `database/` | MySQL 连接池管理 + ORM 模型（`core_memories`、`memory_relations`、`memory_summaries` 表） |
| `memory/` | 记忆系统：抽取（`extract.py`）、检索（`retrieve.py`）、分层归档（`archiver.py`） |
| `llm/` | LLM 推理：小模型前缀 + 大模型续写的流式级联（`inference.py`）、轻量 ReAct 循环（`react.py`） |
| `tools/` | 内置工具注册 + 4 个工具实现（`understand_image`、`understand_audio`、`search_memory`、`analyze_emotion`） |
| `embedding/bge_m3.py` | BGE-M3（768维）文本 Embedding |
| `embedding/whisper.py` | Whisper 音频转录 + 声纹嵌入 |
| `embedding/video_mae.py` | VideoMAE 视频关键帧 Embedding |

---

## 架构变更总结

| 维度 | 修改前 | 修改后 |
|------|--------|--------|
| 核心能力 | 单一向量检索服务 | 完整 AI 对话服务 |
| LLM 对话 | 无 | 支持流式级联推理（小模型前缀 + 大模型续写） |
| 记忆系统 | 无 | 支持记忆分层管理（抽取、检索、分层归档） |
| 工具调用 | 无 | 支持 ReAct 工具调用（图像理解、音频转录、记忆检索、情感分析） |
| 多模态 | 仅文本 Embedding（ChineseCLIP） | 新增 BGE-M3 文本、Whisper 音频、VideoMAE 视频 Embedding |
| 数据持久化 | 无 | 新增 MySQL 持久化记忆存储 |
| API 接口 | 4 个基础接口 | 新增 OpenAI 兼容 `/v1/chat/completions` 流式接口 + 记忆状态接口 |