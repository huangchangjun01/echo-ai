"""Unit tests for utils.tools.VectorSearchTool."""
from __future__ import annotations

from typing import Any


class _FakeStore:
    def __init__(self):
        self.last_where = None
        self.last_n = None

    def query(self, query_text, n_results=5, embedding_fn=None, where=None):
        self.last_where = where
        self.last_n = n_results
        tenant = (where or {}).get("userId", "?")
        return {"ids": [[f"uuid-{tenant}"]], "documents": [[f"doc for {tenant}"]]}


def test_tool_requires_user_id():
    from utils.tools import VectorSearchTool

    store = _FakeStore()
    tool = VectorSearchTool(vector_store=store, embedding_fn=lambda t: [[0.1]])
    import pytest

    with pytest.raises(ValueError, match="user_id"):
        tool._run(query="hi", k=3, user_id="")


def test_tool_passes_user_id_as_filter():
    from utils.tools import VectorSearchTool

    store = _FakeStore()
    tool = VectorSearchTool(vector_store=store, embedding_fn=lambda t: [[0.1]])
    result = tool._run(query="hi", k=3, user_id="alice")
    assert store.last_where == {"userId": "alice"}
    assert store.last_n == 3
    assert result["ids"] == [["uuid-alice"]]


def test_tool_args_schema_marks_user_id_required():
    # Pydantic should reject missing user_id.
    import pytest

    from utils.tools import _VectorSearchInput

    with pytest.raises(Exception):
        _VectorSearchInput(query="hi", k=3)
    # And accept it when supplied.
    parsed = _VectorSearchInput(query="hi", k=3, user_id="u1")
    assert parsed.user_id == "u1"


def test_tool_schemas_expose_function_shape():
    from utils.tools import tool_schemas

    schemas = tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "vector_search"
    params: dict[str, Any] = schemas[0]["function"]["parameters"]
    assert "user_id" in params["properties"]
    assert "user_id" in params["required"], "user_id MUST be required in the schema for LLM function-calling agents"