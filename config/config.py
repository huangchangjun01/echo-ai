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


class Settings(BaseSettings):
    """Top-level settings. Loaded from process env + .env (highest priority: env)."""

    model_config = _BASE_CONFIG

    weaviate: WeaviateSettings = Field(default_factory=WeaviateSettings)
    qiniu: QiniuSettings = Field(default_factory=QiniuSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)

    vector_similarity_threshold: float = Field(0.7, ge=-1.0, le=1.0)


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