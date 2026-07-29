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
    api_key: str | None = None

    # 阈值：CLIP 跨模态 cosine 通常 0.20~0.45，BGE-M3 文本相似 0.5~0.95。
    # 共用 0.7 会把 CLIP 命中全部过滤掉，因此按 collection 分开配置。
    doc_threshold: float = Field(0.25, ge=-1.0, le=1.0)
    memory_threshold: float = Field(0.7, ge=-1.0, le=1.0)

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


class BGEM3Settings(BaseSettings):
    """BGE-M3 文本向量：768 维。失败时回退到 CLIP / sentence-transformers。"""

    model_config = SettingsConfigDict(env_prefix="BGE_M3_", **_BASE_CONFIG)

    model: str = "BAAI/bge-m3"
    dim: int = 768
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
    memory: MemorySettings = Field(default_factory=MemorySettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    app: AppSettings = Field(default_factory=AppSettings)

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