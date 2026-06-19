"""Compatibility shim that exposes get_vector_store backed by Weaviate.

This module mirrors the previous public API so other app modules can continue to import
from app.vector_store import get_vector_store
"""
from vector.vector_store import WeaviateVectorStore, get_vector_store

__all__ = ["WeaviateVectorStore", "get_vector_store"]

