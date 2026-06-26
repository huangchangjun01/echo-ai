from __future__ import annotations

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
    api_key: str | None = None

    def resolved_host_port(self) -> tuple[str, int]:
        """Return (host, port), tolerating a WEAVIATE_HOST that already embeds `:port`.

        Some deployments (and the existing .env in this repo) set
        `WEAVIATE_HOST=host:port`. Treat that as authoritative and fall back to
        the dedicated `WEAVIATE_PORT` only when the host has no port suffix.
        """
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


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", **_BASE_CONFIG)

    model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    device: str = "auto"
    dim: int = 512
    batch_size: int = 8
    chunk_size: int = 256
    chunk_overlap: int = 32
    warmup_on_start: bool = True


class IngestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INGEST_", **_BASE_CONFIG)

    max_download_bytes: int = 50 * 1024 * 1024
    download_timeout_seconds: int = 60
    download_retries: int = 3
    temp_dir: str | None = None
    enable_chunking: bool = True
    cache_ttl_seconds: int = 600
    cache_maxsize: int = 1024


# ── LLM 模型配置（小模型 + 大模型） ────────────────────────────────
class SmallModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SMALL_LLM_", **_BASE_CONFIG)

    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    api_key: str = ""
    max_tokens: int = 64
    temperature: float = 0.7


class LargeModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LARGE_LLM_", **_BASE_CONFIG)

    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.5-plus"
    api_key: str = ""
    max_tokens: int = 2048
    temperature: float = 0.7


class LLMSettings(BaseSettings):
    model_config = _BASE_CONFIG

    small: SmallModelSettings = Field(default_factory=SmallModelSettings)
    large: LargeModelSettings = Field(default_factory=LargeModelSettings)


# ── 记忆系统配置 ──────────────────────────────────────────────────
class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEMORY_", **_BASE_CONFIG)

    # L0 核心记忆：高频、高重要性
    l0_max_count: int = 20
    l0_min_importance: float = 0.8
    # L1 近期记忆：中等重要性
    l1_max_count: int = 100
    l1_min_importance: float = 0.5
    # L2 归档记忆：低重要性
    l2_max_count: int = 500

    # 语义去重阈值（余弦相似度）
    dedup_threshold: float = 0.92
    # 记忆合并阈值（相似度高于此值则合并）
    merge_threshold: float = 0.85
    # 矛盾清理阈值（相似但情感/事实冲突则清理）
    contradiction_threshold: float = 0.80

    # 摘要最大长度
    max_summary_length: int = 500
    # 最大记忆摘要数
    max_summaries: int = 50


# ── 多模态 Embedding 配置 ─────────────────────────────────────────
class BGE3Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BGE_M3_", **_BASE_CONFIG)

    model_name: str = "BAAI/bge-m3"
    dim: int = 768
    batch_size: int = 8
    device: str = "auto"


class CLIPSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLIP_", **_BASE_CONFIG)

    model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16"
    dim: int = 512
    device: str = "auto"


class WhisperSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WHISPER_", **_BASE_CONFIG)

    model_name: str = "openai/whisper-small"
    device: str = "auto"
    language: str = "zh"


class VideoMAESettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIDEO_MAE_", **_BASE_CONFIG)

    model_name: str = "MCG-NJU/videomae-base"
    device: str = "auto"
    frame_sample_rate: int = 10
    max_frames: int = 16


class MultiModalSettings(BaseSettings):
    model_config = _BASE_CONFIG

    bge_m3: BGE3Settings = Field(default_factory=BGE3Settings)
    clip: CLIPSettings = Field(default_factory=CLIPSettings)
    whisper: WhisperSettings = Field(default_factory=WhisperSettings)
    video_mae: VideoMAESettings = Field(default_factory=VideoMAESettings)


# ── 顶层 Settings ─────────────────────────────────────────────────
class Settings(BaseSettings):
    """Top-level settings. Loaded from process env + .env (highest priority: env)."""

    model_config = _BASE_CONFIG

    # ── 数据库配置（直接映射 DB_* 环境变量） ────────────────────────
    db_host: str = Field(default="127.0.0.1", validation_alias="DB_HOST")
    db_port: int = Field(default=3306, validation_alias="DB_PORT")
    db_user: str = Field(default="root", validation_alias="DB_USER")
    db_password: str = Field(default="", validation_alias="DB_PASSWORD")
    db_name: str = Field(default="echo_ai", validation_alias="DB_NAME")

    weaviate: WeaviateSettings = Field(default_factory=WeaviateSettings)
    qiniu: QiniuSettings = Field(default_factory=QiniuSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    multimodal: MultiModalSettings = Field(default_factory=MultiModalSettings)

    vector_similarity_threshold: float = Field(0.7, ge=-1.0, le=1.0)

    @property
    def db_dsn(self) -> str:
        """返回 aiomysql 兼容的 DSN 连接字符串."""
        return f"mysql+aiomysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


_cached: Settings | None = None


def get_settings() -> Settings:
    """Lazy-loaded settings singleton."""
    global _cached
    if _cached is None:
        _cached = Settings()
    return _cached


def reload_settings() -> Settings:
    global _cached
    _cached = Settings()
    return _cached


# Backward-compat constants for legacy imports.
settings = get_settings()
WEAVIATE_URL: str = settings.weaviate.resolved_url()
WEAVIATE_CLASS: str = settings.weaviate.class_name
QINIU_BASE_URL: str = settings.qiniu.base_url
VECTOR_SIMILARITY_THRESHOLD: float = settings.vector_similarity_threshold
DB_DSN: str = settings.db_dsn