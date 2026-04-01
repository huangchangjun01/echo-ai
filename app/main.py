from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.models import compute_embedding, compute_text_embedding
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CLIP Image Embedding Service")

class TextEmbeddingRequest(BaseModel):
    text: str

@app.post("/text-embedding", response_model=dict)
async def create_text_embedding(request: TextEmbeddingRequest):
    """
    接收文本，返回 embedding 向量
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Empty text")

        # 计算 text embedding
        embedding = compute_text_embedding(request.text)

        return {"embedding": embedding}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to process text")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embedding", response_model=dict)
async def create_file_embedding(file: UploadFile = File(...)):
    """
    接收上传的图片文件，返回 embedding 向量
    """
    # 检查文件类型（可选，根据需求放宽）
    # if not file.content_type.startswith("image/"):
    #     raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # 读取文件字节
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        # 计算 embedding
        embedding = compute_embedding(contents)

        return {"embedding": embedding}

    except Exception as e:
        logger.exception("Failed to process image")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}