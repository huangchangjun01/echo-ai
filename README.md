# Echo-AI LangChain Migration

这是一个最小的将项目迁移到 LangChain + Weaviate 的示例实现。

快速开始（Windows PowerShell）:

1. 创建虚拟环境并激活：

   powershell -Command "python -m venv .venv; .\.venv\Scripts\Activate.ps1"

2. 安装依赖：

   powershell -Command "pip install --upgrade pip; pip install -r requirements.txt"

3. 启动服务：

   powershell -Command "python run.py"

4. 测试 API：

   curl -X POST http://127.0.0.1:8000/embed -H "Content-Type: application/json" -d '{"texts": ["你好 世界"]}'

迁移说明：

- 已将文本 embedding 封装为 `app.embeddings.ChineseCLIPEmbeddings`，优先使用原有 `app.models.compute_text_embedding`（如果存在），否则回退到 sentence-transformers。
- 向量库使用 Weaviate (外部服务)，配置通过环境变量或 `config/config.py`，示例持久化路径在 `db/weaviate`（仅作为示例，Weaviate 实际运行为独立服务）。
- 提供基础 HTTP 接口：`/embed`, `/add`, `/search`, `/chat`。后续可以把 Go 的工具逐个迁移为 `langchain.tools`。


