"""Generate a PDF changelog report for uncommitted changes in echo-ai."""
from __future__ import annotations

import os
from datetime import datetime
from fpdf import FPDF


OUTPUT_DIR = r"E:\AIWorking\_HCJun_Work"
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "echo-ai_uncommitted_changes_2026-06-19.pdf")
FONT_DIR = r"C:\Windows\Fonts"
FONT_REGULAR = os.path.join(FONT_DIR, "simhei.ttf")
FONT_BOLD = os.path.join(FONT_DIR, "simhei.ttf")


def add_chinese_font(pdf: FPDF) -> None:
    pdf.add_font("SimHei", "", FONT_REGULAR)
    pdf.add_font("SimHei", "B", FONT_BOLD)


def section_title(pdf: FPDF, text: str) -> None:
    pdf.set_font("SimHei", "B", 14)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 8, text, ln=1)
    pdf.set_draw_color(20, 60, 120)
    pdf.set_line_width(0.4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)


def sub_title(pdf: FPDF, text: str) -> None:
    pdf.set_font("SimHei", "B", 12)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 7, text, ln=1)


def body(pdf: FPDF, text: str) -> None:
    pdf.set_font("SimHei", "", 10.5)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(0, 5.6, text)
    pdf.ln(1)


def bullet(pdf: FPDF, text: str) -> None:
    pdf.set_font("SimHei", "", 10.5)
    pdf.set_text_color(20, 20, 20)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5.6, f"· {text}")


def code_block(pdf: FPDF, text: str) -> None:
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.set_fill_color(245, 245, 245)
    pdf.multi_cell(0, 5, text, fill=True)
    pdf.set_text_color(20, 20, 20)
    pdf.ln(1)


def build() -> None:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    add_chinese_font(pdf)

    # ---------- 封面 ----------
    pdf.add_page()
    pdf.set_font("SimHei", "B", 22)
    pdf.set_text_color(20, 60, 120)
    pdf.ln(40)
    pdf.cell(0, 12, "Echo-AI 项目改动报告", ln=1, align="C")
    pdf.set_font("SimHei", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "未提交代码改动功能点清单", ln=1, align="C")
    pdf.ln(10)
    pdf.set_draw_color(20, 60, 120)
    pdf.set_line_width(0.6)
    pdf.line(40, pdf.get_y(), 170, pdf.get_y())
    pdf.ln(8)

    info = [
        ("仓库", "echo-ai"),
        ("分支", "master"),
        ("报告日期", "2026-06-19"),
        ("对比基准", "最近一次提交 b0d2bdc (Update README.md)"),
        ("改动文件数", "25 个修改 + 4 个新增（tests/、.github/、pyproject.toml、.env.example）"),
        ("新增代码行", "约 +1447 / -617"),
    ]
    pdf.set_font("SimHei", "", 11)
    for k, v in info:
        pdf.set_text_color(60, 60, 60)
        pdf.cell(45, 7, k + "：")
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 7, v, ln=1)
    pdf.ln(6)

    pdf.set_font("SimHei", "B", 12)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 8, "改动概览", ln=1)
    body(
        pdf,
        "本次提交集中将后端升级到生产可用的形态：把 Weaviate v3 客户端替换为 v4 客户端，"
        "全面迁移到 pydantic-settings 配置中心，重写异步下载器并加入 SSRF 防护与大小限制，"
        "把入库流水线从「按扩展名粗分类」改为「libmagic 探测 + 分块 + 批量 embedding」，"
        "并把 LangChain Agent 工具与 FastAPI 接口对齐到「强制 user_id 租户隔离」的安全模型。"
        "配套新增 pytest 测试套件与 GitHub Actions CI 流水线。"
    )

    # ---------- 1. 配置文件 ----------
    pdf.add_page()
    section_title(pdf, "1. 依赖与配置 (requirements.txt / config/config.py / .env.example)")

    sub_title(pdf, "1.1 requirements.txt —— 依赖升级")
    body(
        pdf,
        "新增 / 升级多个核心库，为新的架构提供底层支撑。整体从「单文件、可选依赖」"
        "形态调整为「显式声明 + 平台分支」形态。"
    )
    bullet(pdf, "LangChain 拆分：新增 langchain-core 与 langchain-text-splitters，分别提供 BaseTool 与 RecursiveCharacterTextSplitter。")
    bullet(pdf, "Weaviate 客户端升级：weaviate-client 由 v3 (>=3.0.0) 升级到 v4 (>=4.5.0)。")
    bullet(pdf, "配置中心：引入 pydantic>=2.5.0、pydantic-settings>=2.1.0，替代散落的 os.getenv。")
    bullet(pdf, "异步与重试：httpx>=0.25.0（替代 requests）、tenacity>=8.2.0、cachetools>=5.3.0。")
    bullet(pdf, "MIME 探测：python-magic 跨平台实现，Windows 上使用 python-magic-bin 预编译二进制。")
    bullet(pdf, "测试：新增 pytest>=7.4.0、pytest-asyncio>=0.23.0。")

    sub_title(pdf, "1.2 config/config.py —— pydantic-settings 配置中心")
    body(
        pdf,
        "将原本散落在模块顶层的常量集中到 BaseSettings 子类，并提供懒加载单例，"
        "同时通过模块级常量保留向后兼容。"
    )
    bullet(pdf, "WeaviateSettings：url / host / scheme / port / class_name / api_key；resolved_host_port() 兼容 WEAVIATE_HOST 中带 :port 的旧写法。")
    bullet(pdf, "QiniuSettings：base_url、allowed_subdomains（逗号分隔解析为 list）。")
    bullet(pdf, "EmbeddingSettings：model_name / device / dim / batch_size / chunk_size / chunk_overlap / warmup_on_start。")
    bullet(pdf, "IngestSettings：max_download_bytes / download_timeout_seconds / download_retries / temp_dir / enable_chunking / cache_ttl_seconds / cache_maxsize。")
    bullet(pdf, "顶层 Settings：组合四个子设置，对外暴露 vector_similarity_threshold（带 -1~1 校验）。")
    bullet(pdf, "get_settings() 懒加载 + 模块级 settings 兼容常量（WEAVIATE_URL / WEAVIATE_CLASS / QINIU_BASE_URL / VECTOR_SIMILARITY_THRESHOLD），旧 import 仍可用。")

    sub_title(pdf, "1.3 .env.example —— 配置示例文件")
    body(
        pdf,
        "新增配置模板，列出所有可调项及默认值（Weaviate、Qiniu、Embedding、Ingest、检索阈值），"
        "不再依赖 README 截屏。"
    )

    sub_title(pdf, "1.4 pyproject.toml —— 项目级工具配置")
    bullet(pdf, "[tool.ruff]：line-length=110、target=py310，启用 E/F/I/B/UP/SIM/RUF 规则集，关闭 B008/B904 等。")
    bullet(pdf, "[tool.pytest.ini_options]：asyncio_mode=auto、testpaths=[\"tests\"]、addopts=\"-q\"。")

    # ---------- 2. Embedding ----------
    pdf.add_page()
    section_title(pdf, "2. Embedding 推理 (embedding/models.py / embedding/embeddings.py / embedding/__init__.py)")

    sub_title(pdf, "2.1 embedding/models.py —— 模型层重构")
    bullet(pdf, "线程安全懒加载：_model_lock + _model/_processor/_device 三件套，避免并发启动时重复加载。")
    bullet(pdf, "detect_device(\"auto\")：按 cuda → mps → cpu 顺序选择运行设备。")
    bullet(pdf, "compute_text_embeddings(texts, device)：批量文本 embedding，调用 processor(text=...) 并 L2 归一化。")
    bullet(pdf, "compute_image_embeddings(images, device)：批量图像 embedding，支持 bytes / PIL.Image / 文件路径。")
    bullet(pdf, "warmup(device, batch_size)：在 lifespan 中跑一次 dummy forward，降低首请求延迟。")
    bullet(pdf, "_extract_text_features / _extract_image_features：兼容 transformers 5.x 的 BaseModelOutputWithPooling 与旧版裸 tensor。")
    bullet(pdf, "保留单条 API compute_embedding / compute_text_embedding 作为向后兼容入口。")

    sub_title(pdf, "2.2 embedding/embeddings.py —— LangChain Embeddings 实现")
    bullet(pdf, "基类从 langchain.embeddings.base 切到 langchain_core.embeddings.Embeddings。")
    bullet(pdf, "构造函数读取 get_settings().embedding 注入 model_name / device / dim，保留参数覆盖能力。")
    bullet(pdf, "优先调用 _repo_models.compute_text_embeddings / compute_image_embeddings（批量），回退到 sentence-transformers，最后回退到 SHA256 确定性向量。")
    bullet(pdf, "新增 warmup() 方法包装 _repo_models.warmup。")
    bullet(pdf, "_normalize_batch：统一把 numpy / tensor / 任意可迭代结果归一为 list[list[float]]，长度不足时补零向量。")

    sub_title(pdf, "2.3 embedding/__init__.py —— 公开符号调整")
    body(pdf, "__all__ 扩展到 9 个符号，包含 ChineseCLIPEmbeddings、compute_text_embeddings / compute_image_embeddings、detect_device、warmup 等。")

    # ---------- 3. Vector Store ----------
    pdf.add_page()
    section_title(pdf, "3. 向量库 (vector/vector_store.py)")

    sub_title(pdf, "3.1 整体重写为 REST 客户端")
    body(
        pdf,
        "放弃 weaviate v3 客户端的 gRPC 走法（线上环境通常只暴露 8080），改用 weaviate v4 客户端的"
        "REST 端点自行拼 GraphQL，降低部署耦合。"
    )

    sub_title(pdf, "3.2 _WeaviateHttpClient")
    bullet(pdf, "基于 httpx.Client 实现的薄 REST 封装，覆盖 schema / 对象 / 搜索三类操作。")
    bullet(pdf, "class_exists / get_class / create_class / add_property：幂等的模式管理。")
    bullet(pdf, "batch_insert / get_object / delete_object：批量写与按 ID 读删。")
    bullet(pdf, "graphql(query)：通用 GraphQL 入口。")
    bullet(pdf, "from_settings()：按 WeaviateSettings 拼装 base_url，支持 api_key Bearer 头。")

    sub_title(pdf, "3.3 WeaviateVectorStore")
    bullet(pdf, "新增 userId / chunkIndex 字段；userId 标记为 indexFilterable 用于租户过滤。")
    bullet(pdf, "_ensure_collection / _ensure_filterable_properties：首次创建 class，已存在则比对属性并按需 ALTER（422 视为已存在，幂等）。")
    bullet(pdf, "add_texts：批量写入，返回真实 Weaviate UUID；写入后清空缓存。")
    bullet(pdf, "query(query_text, n_results, embedding_fn, where)：强制要求 where['userId'] 非空，否则抛 ValueError —— 防止越权查询。")
    bullet(pdf, "Top-k 改造：fetch_limit = k*10 → 按 1-distance 计算相似度 → 过阈值 → 排序 → 截前 k，返回 distances 列。")
    bullet(pdf, "TTLCache 缓存：以 (query, k, where) 为 key 缓存查询结果，TTL 与容量从 IngestSettings 注入。")
    bullet(pdf, "get_document / delete / close：仍按 doc_id 读写；delete 后清空缓存。")

    sub_title(pdf, "3.4 进程级单例")
    bullet(pdf, "get_vector_store()：双重检查锁的懒加载单例。")
    bullet(pdf, "reset_vector_store()：测试 / 关闭时显式重置，agent_runner 的 lifespan 会在退出时调用。")

    # ---------- 4. Ingest ----------
    pdf.add_page()
    section_title(pdf, "4. 入库流水线 (biz/ingest.py)")

    sub_title(pdf, "4.1 流程改造")
    body(
        pdf,
        "原流程：按文件扩展名判断文本/图像，串行嵌入一条。\n"
        "新流程：libmagic 探测 MIME → 类型分流 → 文本分块 + 批量 embedding → 批量写入。"
    )
    bullet(pdf, "_detect_mime(data)：优先用 magic.from_buffer(mime=True)，libmagic 不可用时回退到「magic bytes + 可打印字符启发式」识别器。")
    bullet(pdf, "SUPPORTED_IMAGE_MIMES / SUPPORTED_TEXT_MIMES / REJECTED_BINARY_MIMES 三个白/黑名单。")
    bullet(pdf, "_decode_text：依次尝试 utf-8 / utf-8-sig / gbk / gb18030，最后回退到 errors='replace'。")
    bullet(pdf, "_split_text：使用 langchain_text_splitters.RecursiveCharacterTextSplitter（按 \\n\\n / \\n / 中英文句末标点 / 空格 / 字符 切分），失败回退到固定窗口。")
    bullet(pdf, "_download_with_retry：tenacity AsyncRetrying 包装，指数退避（1~10s），重试次数从 IngestSettings 注入。")
    bullet(pdf, "ingest_file(...)：返回 IngestResult(success, file_id, chunks, error) 数据类，便于上层日志与测试。")
    bullet(pdf, "_ingest_text / _ingest_image：分别负责文本分块、批量 compute_text_embeddings（用 asyncio.to_thread 避免阻塞事件循环）以及单条图像 embedding。")
    bullet(pdf, "ID 命名：{file_id}:{chunk_index}，metadata 写入 chunkIndex / totalChunks。")
    bullet(pdf, "保留 ingest_background(...) 旧 API（同步别名，调用 asyncio.run），用于兼容。")

    sub_title(pdf, "4.2 错误处理")
    bullet(pdf, "下载失败：返回 error=\"Download failed: ...\"。")
    bullet(pdf, "MIME 黑名单（mp4/mp3/zip/octet-stream 等）：直接返回 error，不进入 embedding。")
    bullet(pdf, "空文本：error=\"Empty text content\"。")
    bullet(pdf, "embedding / 写库异常：捕获后返回 error，进程不崩溃。")

    # ---------- 5. Downloader ----------
    pdf.add_page()
    section_title(pdf, "5. 下载器 (utils/downloader.py)")

    sub_title(pdf, "5.1 SSRF 防护")
    bullet(pdf, "_ip_is_blocked：拒绝 private / loopback / link-local / multicast / reserved / unspecified 段。")
    bullet(pdf, "_is_safe_url：scheme 必须是 http(s)；未配置白名单时校验解析到的 IP 不在内网段。")
    bullet(pdf, "QINIU_ALLOWED_SUBDOMAINS 留空时按 QINIU_BASE_URL 的 host 自动派生白名单。")
    bullet(pdf, "下载走 streaming，按 Content-Length 预检 + 实际累积字节双重卡 max_bytes。")

    sub_title(pdf, "5.2 API 改造")
    bullet(pdf, "download_file_async(url, *, max_bytes, timeout, allowed_subdomains)：异步流式下载主入口。")
    bullet(pdf, "download_file(url, ...)：asyncio.run 封装的同步入口，保留旧签名。")
    bullet(pdf, "新增 DownloadError 异常，下载大小 / 状态码 / 解析失败统一抛出。")
    bullet(pdf, "httpx.AsyncClient.follow_redirects=True，连接超时 10s、整体超时走 IngestSettings。")

    # ---------- 6. Tools ----------
    pdf.add_page()
    section_title(pdf, "6. LangChain 工具 (utils/tools.py)")

    sub_title(pdf, "6.1 VectorSearchTool 升级")
    bullet(pdf, "基类：langchain.tools.BaseTool → langchain_core.tools.BaseTool。")
    bullet(pdf, "args_schema：使用 pydantic BaseModel 定义 query / k / user_id，对外暴露结构化 JSON Schema。")
    bullet(pdf, "_run / _arun：均强制 user_id 非空，调用 vector_store.query 时把 where={\"userId\": user_id} 注入 —— Agent 无法绕过租户过滤。")
    bullet(pdf, "构造时显式传入 embedding_fn（_EMBEDDINGS.embed_documents），不再依赖模块全局。")

    sub_title(pdf, "6.2 OpenAI 风格 schema")
    bullet(pdf, "tool_schemas()：返回与 OpenAI function-calling 兼容的 JSON Schema，供非 LangChain 的 LLM 客户端直接消费。")
    bullet(pdf, "build_vector_search_tool(vector_store, embedding_fn)：工厂方法。")

    # ---------- 7. FastAPI ----------
    pdf.add_page()
    section_title(pdf, "7. FastAPI 入口 (app/agent_runner.py / app/__init__.py)")

    sub_title(pdf, "7.1 lifespan 重构")
    bullet(pdf, "通过 get_settings() 读取阈值与 warmup 开关。")
    bullet(pdf, "embeddings = ChineseCLIPEmbeddings()；warmup_on_start 时调用 await asyncio.to_thread(embeddings.warmup)，首请求零冷启动。")
    bullet(pdf, "vectorstore 改为每次从 vector.vector_store 模块查找函数（monkeypatch 友好），失败时记录并抛出。")
    bullet(pdf, "依赖（embeddings / vectorstore / settings）挂到 app.state.*，测试中可被覆盖。")
    bullet(pdf, "退出时调用 vs_module.reset_vector_store() 释放 httpx.Client。")

    sub_title(pdf, "7.2 接口变更")
    bullet(pdf, "POST /chat：ChatRequest 强制 user_id（min=1/max=128）、query（min=1/max=2048）、k（1~50）；返回 {query, k, userId, candidates:[{id, document, metadata, distance}]}。")
    bullet(pdf, "POST /search：SearchRequest 的简化别名，内部复用 /chat 逻辑。")
    bullet(pdf, "POST /ingest_file：返回 {ok, queued, fileId}；仍以 BackgroundTasks 异步执行。")
    bullet(pdf, "GET /health：返回 {\"status\":\"ok\"}，未变。")
    bullet(pdf, "新增版本字段 app.version=\"2.0.0\"。")
    bullet(pdf, "chat 失败：ValueError → 400，其它异常 → 500 并 logger.exception。")

    sub_title(pdf, "7.3 app/__init__.py")
    body(pdf, "导入顺序与 __all__ 元素顺序保持字母序，与 ruff（isort）规则一致。")

    # ---------- 8. CI / Tests ----------
    pdf.add_page()
    section_title(pdf, "8. 测试与 CI (tests/、.github/workflows/ci.yml、.claude/settings.local.json)")

    sub_title(pdf, "8.1 tests/ 测试套件")
    bullet(pdf, "conftest.py：把项目根加入 sys.path。")
    bullet(pdf, "test_api.py：用 TestClient + fake embedding / fake vector store 验证 /chat 的 user_id 注入、422 校验、health 探针。")
    bullet(pdf, "test_vector_store.py：通过 _FakeHttpClient 验证 add_texts、query 的 GraphQL where 子句、阈值过滤、缓存行为。")
    bullet(pdf, "test_ingest.py：覆盖文本分块入库与图片入库的关键路径。")
    bullet(pdf, "test_embed.py：ChineseCLIPEmbeddings 的归一化 / 批量 / 回退分支。")
    bullet(pdf, "test_downloader.py：SSRF 防护的子域白名单、IP 黑名单、非 http(s) 拒绝。")
    bullet(pdf, "test_tools.py：VectorSearchTool 必须传 user_id，否则抛 ValueError。")

    sub_title(pdf, "8.2 .github/workflows/ci.yml")
    bullet(pdf, "触发条件：push 到 master/main、pull_request。")
    bullet(pdf, "矩阵：Python 3.10 / 3.11（ubuntu-latest）。")
    bullet(pdf, "步骤：checkout → setup-python（pip 缓存）→ 安装 CPU 版 torch → pip install -r requirements.txt → ruff check . → pytest。")
    bullet(pdf, "为 pytest 注入 Weaviate 端口 env，避免单测访问真实库。")

    sub_title(pdf, "8.3 .claude/settings.local.json")
    body(pdf, "在 Bash 权限白名单追加 \"Bash(ruff check *)\"，允许 lint 命令直接执行。")

    # ---------- 9. README ----------
    pdf.add_page()
    section_title(pdf, "9. 文档 (README.md / run.py)")

    sub_title(pdf, "9.1 README.md 全面更新")
    bullet(pdf, "项目简介明确：Weaviate v4 客户端 + LangChain Embeddings / Tool 抽象 + pydantic-settings 配置。")
    bullet(pdf, "功能表补充：批量 embedding、自动设备选择、MIME 探测 + 分块、缓存、SSRF 防护、tenacity 重试。")
    bullet(pdf, "技术栈补齐：lifespan 预热、langchain-text-splitters、pydantic-settings、httpx、tenacity、python-magic。")
    bullet(pdf, "目录树新增 .github/、pyproject.toml、.env.example、tests/，llm/ 标注为「待扩展」。")
    bullet(pdf, "环境变量：列出 Weaviate、Qiniu、Embedding、Ingest、检索阈值全量项，默认值 0.7。")
    bullet(pdf, "新增「API 速查」小节：/chat、/search、/ingest_file、/health。")
    bullet(pdf, "新增「快速开始」中的 pytest 运行命令与 Docker 启动 Weaviate 的提示。")
    bullet(pdf, "新增「已知未实现」：/chat 暂不接 LLM 总结、未暴露 /metrics、未加鉴权 / CORS。")

    sub_title(pdf, "9.2 run.py")
    body(pdf, "仅做空白行调整，无行为变化。")

    # ---------- 10. 风险 / 注意事项 ----------
    pdf.add_page()
    section_title(pdf, "10. 风险与迁移注意")

    bullet(pdf, "Weaviate 客户端 v3 → v4：Weaviate 服务端建议升级到 1.24+；旧部署可能缺少 userId 字段（脚本会通过 add_property 幂等补齐，但需重启服务）。")
    bullet(pdf, "新 query() 强制 where.userId 非空：旧调用方必须补传，否则抛 ValueError → 400。")
    bullet(pdf, "新增依赖：pydantic-settings、httpx、tenacity、cachetools、python-magic(-bin)、langchain-text-splitters、pytest、pytest-asyncio；requirements.txt 已就绪。")
    bullet(pdf, "下载器加入大小 / IP 校验：自定义对象存储域名若不在 QINIU_ALLOWED_SUBDOMAINS / QINIU_BASE_URL 自动派生的白名单内，将被拒绝下载。")
    bullet(pdf, "embedding 改为批量 + 异步 to_thread：CPU 推理时仍会阻塞事件循环线程，长时间推理建议后续切到独立 worker 池。")
    bullet(pdf, "TTLCache 进程内：多进程部署下各自一份缓存；如需跨实例共享需替换为 Redis。")
    bullet(pdf, "测试使用了 fake REST 客户端与 fake embedding，未集成真实 Weaviate / CLIP；CI 阶段同样不依赖外部服务。")
    bullet(pdf, "AppState.embeddings / vectorstore 现在挂在 app.state 上，单测可 monkeypatch 替换。")

    # ---------- 11. 文件级改动一览 ----------
    pdf.add_page()
    section_title(pdf, "11. 文件级改动一览")

    rows = [
        ("M", ".claude/settings.local.json", "Bash 权限白名单加 ruff check"),
        ("M", "README.md", "全面重写为 v2.0 文档"),
        ("M", "app/__init__.py", "导入与 __all__ 排序"),
        ("M", "app/agent_runner.py", "lifespan / app.state / ChatRequest / 强制 user_id"),
        ("M", "biz/ingest.py", "libmagic 探测 + 分块 + 批量 embedding + tenacity 重试"),
        ("M", "config/config.py", "pydantic-settings 配置中心 + 兼容常量"),
        ("M", "embedding/__init__.py", "__all__ 扩展到 9 个符号"),
        ("M", "embedding/embeddings.py", "LangChain Embeddings + 批量 + warmup + 归一化"),
        ("M", "embedding/models.py", "批量推理 + 设备选择 + warmup + 线程安全懒加载"),
        ("M", "requirements.txt", "依赖升级（v4 客户端 / pydantic-settings / httpx / 测试）"),
        ("M", "run.py", "仅空白调整"),
        ("M", "utils/__init__.py", "导出 DownloadError / 异步下载函数"),
        ("M", "utils/downloader.py", "异步 + SSRF + 大小限制"),
        ("M", "utils/tools.py", "BaseTool + args_schema + 强制 user_id + tool_schemas"),
        ("M", "vector/vector_store.py", "REST 客户端 + 强制租户 + top-k + TTL 缓存"),
        ("A", ".env.example", "配置模板"),
        ("A", ".github/workflows/ci.yml", "GitHub Actions：lint + test 矩阵"),
        ("A", "pyproject.toml", "ruff + pytest 配置"),
        ("A", "tests/conftest.py", "把项目根加入 sys.path"),
        ("A", "tests/test_api.py", "FastAPI 接口单测"),
        ("A", "tests/test_downloader.py", "SSRF 防护单测"),
        ("A", "tests/test_embed.py", "ChineseCLIPEmbeddings 单测"),
        ("A", "tests/test_ingest.py", "入库流水线单测"),
        ("A", "tests/test_tools.py", "VectorSearchTool 单测"),
        ("A", "tests/test_vector_store.py", "Weaviate 封装单测"),
    ]
    pdf.set_font("SimHei", "B", 10.5)
    pdf.set_fill_color(20, 60, 120)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(12, 7, "状态", border=1, align="C", fill=True)
    pdf.cell(85, 7, "文件", border=1, align="L", fill=True)
    pdf.cell(0, 7, "改动摘要", border=1, align="L", fill=True, ln=1)
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("SimHei", "", 10)
    for status, fname, desc in rows:
        pdf.cell(12, 6, status, border=1, align="C")
        pdf.cell(85, 6, fname, border=1)
        pdf.cell(0, 6, desc, border=1, ln=1)
    pdf.ln(4)
    body(pdf, "（状态：A = 新增，M = 修改。其它如 .pyc、__pycache__ 已在 .gitignore 范围内，本次只是 build artifact。）")

    # ---------- 页脚 ----------
    pdf.set_y(-15)
    pdf.set_font("SimHei", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Echo-AI 改动报告 · 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf.output(OUTPUT_PDF)
    print(f"PDF written to: {OUTPUT_PDF}")


if __name__ == "__main__":
    build()
