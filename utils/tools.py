from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from vector.vector_store import WeaviateVectorStore


class _VectorSearchInput(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language query to search for.")
    k: int = Field(5, ge=1, le=50, description="Number of results to return (top-k).")
    user_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Tenant scope. MUST be the calling user's id. Used as the multi-tenant "
            "boundary so the search cannot leak across users."
        ),
    )


class VectorSearchTool(BaseTool):
    """LangChain tool exposing Weaviate semantic search to agents.

    Compatible with both langchain-core `BaseTool` and ReAct / function-calling agents.
    The `user_id` parameter is REQUIRED for tenant isolation.
    """

    name: str = "vector_search"
    description: str = (
        "Search the current user's documents by semantic similarity. "
        "Use this when the user asks about previously uploaded documents or images. "
        "Input: a natural-language query, a user id, and an optional result count."
    )
    args_schema: type[BaseModel] = _VectorSearchInput

    def __init__(self, vector_store: WeaviateVectorStore, embedding_fn):
        super().__init__()
        self._vs = vector_store
        self._embedding_fn = embedding_fn

    def _run(self, query: str, k: int = 5, user_id: str = "", **_: Any) -> dict[str, Any]:
        if not user_id:
            raise ValueError("VectorSearchTool requires user_id for tenant isolation.")
        return self._vs.query(
            query_text=query,
            n_results=k,
            embedding_fn=self._embedding_fn,
            where={"userId": user_id},
        )

    async def _arun(self, query: str, k: int = 5, user_id: str = "", **_: Any) -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self._run, query, k, user_id)


def build_vector_search_tool(
    vector_store: WeaviateVectorStore, embedding_fn
) -> VectorSearchTool:
    return VectorSearchTool(vector_store=vector_store, embedding_fn=embedding_fn)


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style function schema for non-LangChain callers."""
    return [
        {
            "type": "function",
            "function": {
                "name": "vector_search",
                "description": (
                    "Search the current user's documents by semantic similarity. "
                    "Use this when the user asks about previously uploaded documents or images. "
                    "Input: a natural-language query, a user id, and an optional result count."
                ),
                "parameters": _VectorSearchInput.model_json_schema(),
            },
        }
    ]