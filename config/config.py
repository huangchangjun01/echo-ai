from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)


class WeaviateSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WEAVIATE_", **_BASE_CONFIG)

    url: str | None = None
    host: str = "localhost"
    scheme: str = "http"
    port: int = 8080
    class_name: str = Field("EchoDoc", alias="class")
    memory_class: str = Field("EchoMemory", alias="memoryClass")
    # 回忆记忆向量集合（独立于 EchoDoc / EchoMemory）
    recall_class: str = Field("EchoRecall", alias="recallClass")
    api_key: str | None = None

    # 阈值：CLIP 跨模态 cosine 通常 0.20~0.45，BGE-M3 文本相似 0.5~0.95。
    # 共用 0.7 会把 CLIP 命中全部过滤掉，因此按 collection 分开配置。
    doc_threshold: float = Field(0.25, ge=-1.0, le=1.0)
    memory_threshold: float = Field(0.7, ge=-1.0, le=1.0)
    recall_threshold: float = Field(0.5, ge=-1.0, le=1.0)

    def resolved_host_port(self) -> tuple[str, int]:
        host = (self.host or "").strip()
        if "@" in host:
            host = host.rsplit("@", 1)[-1]
        if host.startswith("["):
            end = host.find("]")
            if end != -1:
                host_only = host[1:end]
                port_str = host[end + 2 :] if end + 1 < len(host) and host[end + 1] == ":" else ""
                return host_only, int(port_str) if port_str else self.port
        if ":" in host:
            host_only, _, port_str = host.rpartition(":")
            try:
                return host_only, int(port_str)
            except ValueError:
                return host_only, self.port
        return host, self.port

    def resolved_url(self) -> str:
        if self.url:
            return self.url
        host, port = self.resolved_host_port()
        return f"{self.scheme}://{host}:{port}"


class QiniuSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QINIU_", **_BASE_CONFIG)

    base_url: str = ""
    allowed_subdomains: list[str] = Field(default_factory=list)
    # 上传/删除凭证（与 echo-core 一致；echo-ai 直传/删除对象）
    access_key: str = ""
    secret_key: str = ""
    bucket_name: str = ""

    @field_validator("allowed_subdomains", mode="before")
    @classmethod
    def _split(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v


class DBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_", **_BASE_CONFIG)

    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = ""
    name: str = "echo_ai"
    pool_min: int = 1
    pool_max: int = 10
    # 空闲连接最长存活秒数。远端 MySQL / NAT / 防火墙会静默断开长时间空闲的 TCP 连接，
    # 而 aiomysql 只能识别"已收到 FIN/RST"的连接，半开连接会被原样发出去，
    # 导致下一条 SQL 一直阻塞到 OS 超时（Windows: WinError 121 信号灯超时时间已到）。
    # 取值需明显小于 MySQL 的 wait_timeout 以及链路上 NAT 的空闲回收时间。
    pool_recycle: int = 300
    # 建连超时（秒）：避免网络异常时 connect 无限期挂起。
    connect_timeout: int = 10

    def resolved_dsn(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "db": self.name,
        }


class LLMSettings(BaseSettings):
    """OpenAI 兼容 LLM 配置：大模型主用 + 情感微模型（小模型）前缀。"""

    model_config = SettingsConfigDict(env_prefix="LLM_", **_BASE_CONFIG)

    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""

    # 大模型
    model: str = "gpt-4o-mini"
    max_tokens: int = 2048
    temperature: float = 0.7

    # 小模型（情感微模型 / 快速前缀）
    small_model: str = "gpt-4o-mini"
    small_max_tokens: int = 64
    small_temperature: float = 0.5
    small_base_url: str = ""
    small_api_key: str = ""

    def small_resolved(self) -> tuple[str, str]:
        base = self.small_base_url or self.base_url
        key = self.small_api_key or self.api_key
        return base, key


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", **_BASE_CONFIG)

    model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    device: str = "auto"
    dim: int = 512
    batch_size: int = 8
    chunk_size: int = 256
    chunk_overlap: int = 32
    warmup_on_start: bool = True
    # HuggingFace 镜像，国内部署需要切到 hf-mirror.com，否则直连
    # huggingface.co 会在 transformers 探测 PEFT adapter 的 HEAD 请求阶段 SSL 失败。
    endpoint: str | None = None
    # 强制只读本地缓存（模型已下到 ~/.cache/huggingface 时用）
    local_files_only: bool = False
    # 单次 HF 拉取超时（秒），避免在镜像不通时拉满 50s 重试窗口
    download_timeout: int = 15


class BGEM3Settings(BaseSettings):
    """BGE-M3 文本向量：768 维。失败时回退到 CLIP / sentence-transformers。"""

    model_config = SettingsConfigDict(env_prefix="BGE_M3_", **_BASE_CONFIG)

    model: str = "BAAI/bge-m3"
    dim: int = 768
    device: str = "auto"
    # 镜像端点：参见 EmbeddingSettings.endpoint
    endpoint: str | None = None
    local_files_only: bool = False
    download_timeout: int = 15


class WhisperSettings(BaseSettings):
    """本地音频转录（openai-whisper）。

    `model` 是中文识别质量的**决定性因素**。曾经写死的 `base`（74M 多语种）在中文上
    同音字选错极为严重——实测把"上吐下泻"识别成"上土下泄"、"一整晚"识别成"一整碗"。
    档位对照（10s 音频、CPU 实测）：

        base    1.7s  / ~0.15G  错别字多，不可用于中文
        small   2.7s  / ~0.5G   基本正确
        medium  9.2s  / ~1.5G   稳定正确（默认）
        turbo   8.4s  / ~0.9G   实测反而不如 small，且加载极慢，不推荐

    解析是后台任务，用户对延迟不敏感，故默认取 medium 换质量。
    显存/内存吃紧时可用 `WHISPER_MODEL=small` 降档。
    """

    model_config = SettingsConfigDict(env_prefix="WHISPER_", **_BASE_CONFIG)

    model: str = "medium"
    # 显式指定语种可省掉一次语种自动检测（实测省约 1.4s），也避免短音频/嘈杂时
    # 误判语种导致输出乱码。留空则回退为自动检测。
    language: str = "zh"
    # whisper 中文倾向输出繁体（small 以上尤其明显）。开启后用 opencc 做识别后的
    # 确定性繁转简。注意：不要改用 initial_prompt 引导简体——实测那会干扰选字，
    # 把"上吐下泻"带偏成"上吐下泄"。
    simplified: bool = True
    device: str = "auto"


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEMORY_", **_BASE_CONFIG)

    l0_limit: int = 20
    l1_topk: int = 8
    l2_topk: int = 20
    dedup_threshold: float = 0.92
    react_max_iter: int = 3
    enable_async_extract: bool = True
    # 级联前缀：小模型先吐的 token 数（首字延迟敏感）。太小看不到预览，太大拖慢首屏。
    cascade_prefix_tokens: int = 64
    # 单次搜索最多返回的命中数；默认 1，UI 只展示最相关的一条。
    search_max_hits: int = 1

    # 意图识别闸门：每次 chat 先用小模型分类（chat/recall/text_search/image_search/doc_search），
    # 下游 RAG / 工具调度都按这个意图走。给定当前默认小模型是推理型，<2.5s
    # 几乎不可能完成「思考 + JSON」一轮；给到 4s 让 90%+ 一次响应能完成。
    # 若换成非推理小模型，把这个值调回 1500 即可。
    intent_classifier_enabled: bool = True
    intent_timeout_ms: int = 40000


class IngestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INGEST_", **_BASE_CONFIG)

    max_download_bytes: int = 50 * 1024 * 1024
    download_timeout_seconds: int = 60
    download_retries: int = 3
    temp_dir: str | None = None
    enable_chunking: bool = True
    cache_ttl_seconds: int = 600
    cache_maxsize: int = 1024


class ModelCacheSettings(BaseSettings):
    """本地模型缓存目录配置（MODEL_ 前缀）。

    统一管理语音 / 视频 / 文本 / 图片等本地模型：
    - 每次启动 / 使用优先从指定缓存目录获取，取不到时才下载。
    - 结构：<cache_dir>/huggingface/hub（HF 模型）、<cache_dir>/whisper（Whisper）。
    - 留空时回退到各库默认缓存位置（HF: ~/.cache/huggingface，whisper: ~/.cache/whisper），
      保证已缓存模型无需重复下载。
    """

    model_config = SettingsConfigDict(env_prefix="MODEL_", **_BASE_CONFIG)

    # 指定模型缓存根目录；空 = 使用各库默认位置。
    cache_dir: str | None = None
    # 缓存未命中时是否允许联网下载。false = 仅用缓存（离线部署）。
    auto_download: bool = True


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", **_BASE_CONFIG)

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


class Settings(BaseSettings):
    """顶级配置中心：从 .env + 进程环境变量加载（env 优先）。"""

    model_config = _BASE_CONFIG

    weaviate: WeaviateSettings = Field(default_factory=WeaviateSettings)
    qiniu: QiniuSettings = Field(default_factory=QiniuSettings)
    db: DBSettings = Field(default_factory=DBSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    bge_m3: BGEM3Settings = Field(default_factory=BGEM3Settings)
    whisper: WhisperSettings = Field(default_factory=WhisperSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    app: AppSettings = Field(default_factory=AppSettings)
    model: ModelCacheSettings = Field(default_factory=ModelCacheSettings)

    vector_similarity_threshold: float = Field(0.7, ge=-1.0, le=1.0)
    # 兼容旧字段：doc 端阈值
    doc_similarity_threshold: float | None = None


_cached: Settings | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """延迟初始化的 settings 单例（lru_cache 保证进程内唯一）。"""
    global _cached
    if _cached is None:
        _cached = Settings()
    return _cached


def reload_settings() -> Settings:
    global _cached
    _cached = Settings()
    get_settings.cache_clear()
    return _cached


settings = get_settings()
WEAVIATE_URL: str = settings.weaviate.resolved_url()
WEAVIATE_CLASS: str = settings.weaviate.class_name
QINIU_BASE_URL: str = settings.qiniu.base_url
VECTOR_SIMILARITY_THRESHOLD: float = settings.vector_similarity_threshold