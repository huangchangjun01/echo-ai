# Echo-AI

一个基于 **Chinese-CLIP + Weaviate + LangChain** 的多模态 RAG（检索增强生成）服务，提供文本/图像的向量化、向量检索与文件入库能力。

> 项目源码与提交记录中保留的旧说明仅供迁移参考；当前实现已切换为 Weaviate 作为唯一向量库，并使用 LangChain 风格的 Embeddings / Tool 抽象。

---

## 1. 业务功能

| 功能 | 说明 |
| --- | --- |
| 多模态 Embedding | 基于 `OFA-Sys/chinese-clip-vit-base-patch16`，对中文文本与图像生成同一向量空间下的表征 |
| 文本/图像入库 | 通过七牛云对象存储下载文件，按扩展名分流到文本或图像 embedding，写入 Weaviate |
| 相似度检索 | 余弦相似度检索，按阈值过滤后返回最相关的一条结果 |
| RAG 对话接口 | `/chat` 暴露给上游业务，按 query 召回上下文（后续可接入 LLM 生成答案） |
| Agent 工具 | `utils/tools.py` 中封装了 `VectorSearchTool`，便于后续接入 LangChain Agent |
| 健康检查 | `/health` 端点，便于存活/就绪探针 |

---

## 2. 技术栈

- **Web 框架**：FastAPI 0.104 + Uvicorn 0.24
- **多模态模型**：Chinese-CLIP（`OFA-Sys/chinese-clip-vit-base-patch16`，HuggingFace Transformers + PyTorch）
- **向量库**：Weaviate（v3 client，使用 client-side embedding）
- **LLM 接入**（配置项）：Qwen（DashScope）、SiliconFlow 等 OpenAI 兼容接口
- **对象存储**：七牛云（通过 `QINIU_BASE_URL` + `fileKey` 拼装下载链接）
- **工具链**：LangChain、sentence-transformers、python-dotenv、requests

---

## 3. 目录结构

```
echo-ai/
├── app/
│   └── agent_runner.py        # FastAPI 入口，lifespan 内初始化 embedding & vector store
├── biz/
│   └── ingest.py              # 后台入库任务：下载文件 → 解析 → embedding → 写向量库
├── config/
│   └── config.py              # 统一从 .env 加载配置
├── embedding/
│   ├── embeddings.py          # LangChain Embeddings 实现（ChineseCLIPEmbeddings）
│   └── models.py              # Chinese-CLIP 模型加载与推理
├── llm/                       # LLM 接入预留目录（当前为空，待扩展）
├── utils/
│   ├── downloader.py          # URL 下载工具（基于 requests）
│   └── tools.py               # LangChain BaseTool 封装（VectorSearchTool）
├── vector/
│   └── vector_store.py        # Weaviate 向量库封装
├── tests/                     # 预留测试目录
├── run.py                     # uvicorn 启动入口（兼容 PyCharm Debug）
├── requirements.txt
└── .env                       # 运行配置（含密钥，请勿提交到公开仓库）
```

---

## 4. 环境与配置

`.env` 中的关键项：

```ini
# Weaviate
WEAVIATE_HOST=121.43.145.179:8080
WEAVIATE_SCHEME=http
WEAVIATE_PORT=8080
WEAVIATE_CLASS=EchoDoc

# 七牛云对象存储
QINIU_BASE_URL=tfpdkiq9g.hn-bkt.clouddn.com

# LLM（当前默认走 DashScope 兼容的 Qwen）
LLM_PROVIDER=qwen
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.5-plus
LLM_API_KEY=...
DASHSCOPE_API_KEY=...

# 检索阈值（余弦相似度，范围 [-1, 1]，默认 0.7）
VECTOR_SIMILARITY_THRESHOLD=0.3
```

> `WEAVIATE_URL` 也可以整体给定；若同时提供，`config.config` 会优先解析 `WEAVIATE_URL`。
> `VECTOR_SIMILARITY_THRESHOLD` 通过 `1 - distance` 计算余弦相似度，命中阈值才进入返回列表。

---

## 5. 快速开始

> 下面示例使用 Windows PowerShell，其他平台可自行替换激活命令。

```powershell
# 1. 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 3. 准备 .env（参考上文）

# 4. 启动服务
python run.py
# 监听 0.0.0.0:8000；run.py 会自动检测 PyCharm Debug 模式
```

