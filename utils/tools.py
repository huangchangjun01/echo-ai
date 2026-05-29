from typing import Any, Dict, List
from langchain.tools import BaseTool


class VectorSearchTool(BaseTool):
    name = "vector_search"
    description = "Use this tool to search documents by semantic similarity. Input: {query: str, k: int}. Returns top-k documents."

    def __init__(self, vector_store):
        super().__init__()
        self.vs = vector_store

    def _run(self, query: Dict[str, Any]):
        q = query.get("query") if isinstance(query, dict) else query
        k = int(query.get("k", 5)) if isinstance(query, dict) else 5
        res = self.vs.query(q, n_results=k)
        return res

    async def _arun(self, query: Dict[str, Any]):
        return self._run(query)

