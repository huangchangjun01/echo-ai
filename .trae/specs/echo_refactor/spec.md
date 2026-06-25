# Echo Refactor Spec

## Why
当前系统业务划分不合理，仅提供基础的向量检索和文件入库能力。需要重构为一个支持分层记忆系统、多模态 embedding、轻量 ReAct 对话循环的完整 AI 对话服务。

## What Changes
- **新增**对话接口 `/chat/completions` 支持完整的对话流程，包含预注入记忆、工具调用、异步记忆抽取
- **新增**记忆系统模块 `memory/`，包含记忆抽取（extract）、记忆检索（retrieve）、分层归档（L0/L1/L2）
- **新增**MySQL 记忆存储层，支持核心记忆存储和记忆关系查询
- **新增**多模态理解工具：understand_image（视觉模型）、understand_audio（语音模型）、analyze_emotion（情感分析）
- **增强**Embedding 服务，支持文本（BGE-M3 768维）、图片（CLIP）、音频（Whisper）、视频（VideoMAE/关键帧CLIP）
- **新增**模型推理模块 `llm/`，实现小模型快速前缀生成 + 大模型深度续写的流式级联推理
- **新增**配置：MySQL 连接配置、LLM 模型配置（小模型/大模型）、记忆分层配置
- **重构**现有模块目录结构，使业务职责更清晰
- **BREAKING**: 原 `/chat` 接口保留但功能改变，仅用于向量检索；新增 `/v1/chat/completions` 用于完整对话生成

## Impact
- Affected capabilities: 对话生成、记忆管理、多模态理解、模型推理
- Affected code:
  - `app/agent_runner.py` - 新增对话接口，扩展生命周期初始化
  - `config/config.py` - 新增 MySQL、LLM、记忆系统配置
  - `embedding/` - 扩展支持多模态 embedding（BGE-M3、Whisper、VideoMAE）
  - **新增**: `memory/` - 记忆抽取、检索、分层归档
  - **新增**: `llm/` - LLM 推理、流式级联、工具调用解析
  - **新增**: `tools/` - 内置工具实现（understand_image, understand_audio, search_memory, analyze_emotion）
  - **新增**: `database/` - MySQL 连接和模型定义
  - `.env` - 新增配置项

## ADDED Requirements

### Requirement: Enhanced Dialogue Interface
系统 SHALL 提供增强的对话接口，满足：
- 预注入 L0 核心记忆 + 人格设定到对话 prompt
- 实现轻量 ReAct 循环，调用 LLM 进行推理
- 支持小模型前缀生成 + 大模型续写的流式级联推理
- 检测并执行工具调用：
  - `understand_image`: 进程内视觉模型理解图片内容
  - `understand_audio`: 进程内语音模型理解音频内容
  - `search_memory`: 结合 Weaviate + MySQL 检索用户记忆
  - `analyze_emotion`: 进程内情感分析模型分析对话情感
- 异步入库：对话完成后异步抽取记忆并进行分层归档

#### Scenario: Normal Conversation
- **WHEN** 用户发送对话请求，包含历史消息
- **THEN** 系统预注入 L0 核心记忆，执行 ReAct 循环，按需调用工具，返回流式生成的回复，最后异步抽取记忆归档

### Requirement: Memory System
系统 SHALL 实现三层记忆系统：
- **记忆抽取**:
  - 使用 LLM 抽取原子事实和因果关系
  - 按类型向量化（文本→BGE-M3，图片→CLIP，音频→Whisper+声纹）存入 Weaviate
  - 基于向量相似度 + LLM 进行语义去重
  - 识别记忆关系：causes/update/contradict/extend
  - 情感标注：emotion_tag + intensity
  - 分层归档到 L0/L1/L2 存入 MySQL
  - 相似记忆合并 + 矛盾清理
  - 生成记忆摘要

- **记忆检索**:
  - L0: MySQL 全量加载核心记忆
  - L1: Weaviate 向量检索 Top-K 相关记忆
  - 因果链: MySQL memory_relations 查询关联记忆链
  - 多模态: Weaviate 跨模态检索

### Requirement: Model Inference
系统 SHALL 支持流式级联推理：
- 情感微模型（小模型）快速生成前缀（首字响应 < 200ms）
- 大模型进行深度续写和记忆整合
- 流式输出：小模型前缀 + 大模型续写对用户呈现连贯回复

### Requirement: Embedding Service
系统 SHALL 支持多模态 Embedding：
- 文本: BGE-M3（768 维）
- 图片: CLIP（复用现有 Chinese-CLIP）
- 音频: Whisper 转录 + 声纹嵌入
- 视频: VideoMAE 或关键帧提取后 CLIP 嵌入

### Requirement: MySQL Configuration
系统 SHALL 使用以下 MySQL 配置：
```
DB_HOST=121.43.145.179
DB_PORT=3306
DB_USER=root
DB_PASSWORD=10010hcj
DB_NAME=huangchangjun
```

## MODIFIED Requirements
### Requirement: Configuration
所有模型相关配置 SHALL 集中放在 `.env` 文件中，通过 pydantic-settings 加载。

## REMOVED Requirements
无。保留现有向量检索和文件入库能力，在此基础上扩展新功能。
