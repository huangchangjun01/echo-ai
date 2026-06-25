FROM python:3.10-slim

WORKDIR /app

# 系统依赖：libmagic 用于文件类型检测，ffmpeg 用于视频处理
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（分层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY . .

# 创建模型缓存目录，避免 HuggingFace 缓存到 /root
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

EXPOSE 8000

CMD ["python", "run.py"]