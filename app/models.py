import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

# 全局变量，避免重复加载模型
_model = None
_processor = None


def load_model():
    """加载 CLIP 模型和处理器（首次调用时加载）"""
    global _model, _processor
    if _model is None:
        logger.info("Loading Chinese-CLIP model...")
        # 使用支持中文的 Chinese-CLIP 模型
        model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
        _model = ChineseCLIPModel.from_pretrained(model_name)
        _processor = ChineseCLIPProcessor.from_pretrained(model_name)
        _model.eval()  # 设置为评估模式
        # 如果有 GPU，将模型移到 GPU
        if torch.cuda.is_available():
            _model = _model.cuda()
            logger.info("Model moved to GPU")
        logger.info("Model loaded successfully")
    return _model, _processor


def compute_embedding(image_bytes: bytes) -> list:
    """
    输入图片二进制数据，返回归一化后的 embedding 向量（list of float）
    """
    try:
        # 加载模型（如果尚未加载）
        logger.info("Load embedding model start")
        model, processor = load_model()
        logger.info("Load embedding model end")

        # 将字节数据转换为 PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        logger.info("convert image to PIL-Image")

        # 预处理图片
        logger.info(f"Image size: {image.size}")
        inputs = processor(images=image, return_tensors="pt")
        logger.info("processor image success!")

        # 将输入数据移动到与模型相同的设备
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            # 获取图片特征：get_image_features 直接返回对应向量
            output = model.get_image_features(**inputs)  # 直接获取图片特征，避免不必要的计算
            image_features = output.pooler_output  # 提取池化后的特征向量

        # 归一化（可选，但推荐）
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 转换为 Python list
        embedding = image_features.squeeze().cpu().tolist()
        return embedding

    except Exception as e:
        logger.error(f"Error computing embedding: {e}")
        raise


def compute_text_embedding(text: str) -> list:
    """
    输入文本字符串，返回归一化后的 embedding 向量（list of float）
    """
    try:
        # 加载模型（如果尚未加载）
        logger.info("Load text embedding model start")
        model, processor = load_model()
        logger.info("Load text embedding model end")

        # 预处理文本
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        logger.info("processor text success!")

        # 将输入数据移动到与模型相同的设备
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            # 获取文本特征：get_text_features 直接返回对应向量
            output = model.get_text_features(**inputs)  # 注意：CLIP 模型通常使用 get_image_features 来获取文本特征，因为它们共享同一空间
            text_features = output.pooler_output

        # 归一化（推荐）
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 转换为 Python list
        embedding = text_features.squeeze().cpu().tolist()
        return embedding

    except Exception as e:
        logger.error(f"Error computing text embedding: {e}")
        raise
