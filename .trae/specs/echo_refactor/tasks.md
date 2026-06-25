# Tasks

- [x] Task 1: Git 分支管理
  - [x] SubTask 1.1: 从 master 拉取 echo_refactor 分支
  - [x] SubTask 1.2: 切换到 echo_refactor 分支

- [x] Task 2: 配置层重构 - 扩展 config.py 和 .env
  - [x] SubTask 2.1: 在 config.py 中新增 MySQL 配置（DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME）
  - [x] SubTask 2.2: 在 config.py 中新增 LLM 配置（小模型和大模型各自的 base_url, model, api_key）
  - [x] SubTask 2.3: 在 config.py 中新增记忆系统配置（分层阈值、去重阈值、摘要配置）
  - [x] SubTask 2.4: 在 config.py 中新增 Embedding 多模态配置（BGE-M3, CLIP, Whisper, VideoMAE）
  - [x] SubTask 2.5: 更新 .env 和 .env.example 文件，添加所有新增配置项

- [x] Task 3: 数据库层 - 创建 database/ 模块
  - [x] SubTask 3.1: 创建 database/__init__.py
  - [x] SubTask 3.2: 创建 database/mysql.py，实现 MySQL 连接池管理、异步会话
  - [x] SubTask 3.3: 创建 database/models.py，定义记忆表（core_memories, memory_relations, memory_summaries）的 SQLAlchemy 模型

- [x] Task 4: 记忆系统 - 创建 memory/ 模块
  - [x] SubTask 4.1: 创建 memory/__init__.py
  - [x] SubTask 4.2: 创建 memory/extract.py，实现记忆抽取逻辑（LLM 抽取原子事实+因果关系、向量化、语义去重、关系识别、情感标注、分层归档、合并清理、摘要生成）
  - [x] SubTask 4.3: 创建 memory/retrieve.py，实现记忆检索逻辑（L0 MySQL 全量加载、L1 Weaviate Top-K 检索、因果链查询、跨模态检索）
  - [x] SubTask 4.4: 创建 memory/archiver.py，实现分层归档逻辑（L0/L1/L2 判定与写入 MySQL）

- [x] Task 5: LLM 推理模块 - 创建 llm/ 模块
  - [x] SubTask 5.1: 创建 llm/__init__.py
  - [x] SubTask 5.2: 创建 llm/inference.py，实现小模型快速前缀生成、大模型深度续写、流式级联输出
  - [x] SubTask 5.3: 创建 llm/react.py，实现轻量 ReAct 循环（LLM 调用 + 工具调用解析 + 结果聚合）

- [x] Task 6: 内置工具 - 创建 tools/ 模块
  - [x] SubTask 6.1: 创建 tools/__init__.py
  - [x] SubTask 6.2: 创建 tools/understand_image.py，实现进程内视觉模型图片理解
  - [x] SubTask 6.3: 创建 tools/understand_audio.py，实现进程内语音模型音频理解
  - [x] SubTask 6.4: 创建 tools/search_memory.py，实现结合 Weaviate + MySQL 的记忆搜索工具
  - [x] SubTask 6.5: 创建 tools/analyze_emotion.py，实现进程内情感分析工具

- [x] Task 7: Embedding 服务扩展 - 增强 embedding/ 模块
  - [x] SubTask 7.1: 新增 embedding/bge_m3.py，实现 BGE-M3（768维）文本 Embedding
  - [x] SubTask 7.2: 新增 embedding/whisper.py，实现 Whisper 音频转录 + 声纹嵌入
  - [x] SubTask 7.3: 新增 embedding/video_mae.py，实现 VideoMAE 视频 Embedding（关键帧提取 + CLIP）
  - [x] SubTask 7.4: 更新 embedding/__init__.py，统一导出多模态 embedding 接口

- [x] Task 8: 对话接口重构 - 重构 app/agent_runner.py
  - [x] SubTask 8.1: 新增 `/v1/chat/completions` 接口，实现预注入记忆 + ReAct 循环 + 流式响应
  - [x] SubTask 8.2: 保留原有 `/chat` 接口（向量检索）和 `/search` 别名
  - [x] SubTask 8.3: 扩展 lifespan 生命周期，初始化 MySQL 连接池、加载配置、初始化记忆模块
  - [x] SubTask 8.4: 实现对话完成后异步记忆抽取和分层归档

- [x] Task 9: 更新依赖配置文件
  - [x] SubTask 9.1: 更新 requirements.txt，添加 MySQL（aiomysql/SQLAlchemy）、Whisper、VideoMAE 等依赖
  - [x] SubTask 9.2: 更新 pyproject.toml 配置（如需要）

- [x] Task 10: 生成重构报告
  - [x] SubTask 10.1: 生成报告文件 refactor_report.md
  - [x] SubTask 10.2: 将报告保存在项目根目录 /workspace/refactor_report.md

- [x] Task 11: 服务启动与自测
  - [x] SubTask 11.1: 启动服务，检查是否有启动报错并修复
  - [x] SubTask 11.2: 自动测试 `/health` 接口
  - [x] SubTask 11.3: 自动测试 `/v1/chat/completions` 接口（简单对话）
  - [x] SubTask 11.4: 自动测试原有 `/chat` 接口
  - [x] SubTask 11.5: 自测成功后关闭服务，避免端口占用

- [x] Task 12: Git 提交与推送
  - [x] SubTask 12.1: 提交所有改动到 echo_refactor 分支
  - [x] SubTask 12.2: Push 到远程 echo_refactor 分支

# Task Dependencies
- Task 2 依赖 Task 1（需要在分支上修改）
- Task 3 依赖 Task 2（需要 MySQL 配置）
- Task 4, 5, 6, 7 依赖 Task 2（需要配置），其中 Task 4 依赖 Task 3（需要 MySQL 模型）
- Task 6 依赖 Task 4（search_memory 工具依赖记忆检索）
- Task 8 依赖 Task 4, 5, 6（对话接口需要记忆、LLM、工具模块）
- Task 9, 10 可与其他任务并行
- Task 11 依赖 Task 1-10 全部完成
- Task 12 依赖 Task 11