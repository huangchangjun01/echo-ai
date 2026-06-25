from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class CoreMemory(Base):
    __tablename__ = "core_memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), index=True, nullable=False)
    content = Column(Text, nullable=False)
    memory_type = Column(String(32), default="text")  # text/image/audio/video
    layer = Column(String(8), default="L1")  # L0/L1/L2
    importance = Column(Float, default=0.5)
    emotion_tag = Column(String(32), nullable=True)
    emotion_intensity = Column(Float, default=0.0)
    weaviate_uuid = Column(String(64), nullable=True)
    embedding_dim = Column(Integer, default=512)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


class MemoryRelation(Base):
    __tablename__ = "memory_relations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(Integer, ForeignKey("core_memories.id"), nullable=False)
    target_id = Column(Integer, ForeignKey("core_memories.id"), nullable=False)
    relation_type = Column(String(32), nullable=False)  # causes/update/contradict/extend
    confidence = Column(Float, default=0.5)
    created_at = Column(DateTime, server_default=func.now())


class MemorySummary(Base):
    __tablename__ = "memory_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), index=True, nullable=False)
    summary = Column(Text, nullable=False)
    memory_ids = Column(Text, nullable=True)  # comma-separated core_memory ids
    created_at = Column(DateTime, server_default=func.now())