import os

try:
    # optional - load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Weaviate connection settings - prefer environment variables
# Support multiple ways to configure Weaviate endpoint
# 1) Full URL via WEAVIATE_URL
# 2) Or provide WEAVIATE_SCHEME, WEAVIATE_HOST, WEAVIATE_PORT
WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "")
if not WEAVIATE_URL:
    host = os.getenv("WEAVIATE_HOST", "localhost")
    scheme = os.getenv("WEAVIATE_SCHEME", "http")
    port = os.getenv("WEAVIATE_PORT", "8080")
    # If host already contains port, extract and use it
    if ":" in host:
        parts = host.rsplit(":", 1)
        host = parts[0]
        port = parts[1] if len(parts) > 1 else port
    WEAVIATE_URL = f"{scheme}://{host}:{port}"

# Default class name for storing documents in Weaviate (will be sanitized)
WEAVIATE_CLASS: str = os.getenv("WEAVIATE_CLASS", "EchoDoc")
# Qiniu (七牛云) base URL
QINIU_BASE_URL: str = os.getenv("QINIU_BASE_URL", "tfpdkiq9g.hn-bkt.clouddn.com")

# Vector similarity threshold (cosine similarity)
# 余弦相似度阈值，取值范围 [-1, 1]，业界常规使用 0.7-0.8 作为阈值
# 值越高表示要求越相似，默认 0.7
VECTOR_SIMILARITY_THRESHOLD: float = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", "0.7"))
