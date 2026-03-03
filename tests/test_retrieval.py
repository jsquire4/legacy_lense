"""Tests for the retrieval service (mocked, no API/Qdrant calls)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.retrieval import _extract_routine_name, retrieve


def test_extract_routine_name_found():
    """_extract_routine_name extracts LAPACK routine from query."""
    assert _extract_routine_name("What is DGESV?") == "DGESV"
    assert _extract_routine_name("Explain DGETRF") == "DGETRF"
    assert _extract_routine_name("How does DGEMM work?") == "DGEMM"


def test_extract_routine_name_not_found():
    """_extract_routine_name returns None when no match."""
    assert _extract_routine_name("How does LU decomposition work?") is None
    assert _extract_routine_name("no match here") is None


def test_extract_routine_name_sdczi_prefix():
    """_extract_routine_name only matches S/D/C/Z/I prefix."""
    assert _extract_routine_name("What is DGESV?") == "DGESV"
    assert _extract_routine_name("What is ZGEMM?") == "ZGEMM"


@patch("app.services.retrieval.async_search", new_callable=AsyncMock)
@patch("app.services.retrieval.async_search_by_name", new_callable=AsyncMock)
@patch("app.services.retrieval.embed_query", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_retrieve_embed_failure(mock_embed, mock_search_by_name, mock_search):
    """retrieve returns failed strategy when embed_query returns empty."""
    mock_embed.return_value = []

    result = await retrieve("query", top_k=5)
    assert result["chunks"] == []
    assert result["retrieval_strategy"] == "failed"
    mock_search.assert_not_called()


@patch("app.services.retrieval.async_search", new_callable=AsyncMock)
@patch("app.services.retrieval.async_search_by_name", new_callable=AsyncMock)
@patch("app.services.retrieval.embed_query", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_retrieve_name_match(mock_embed, mock_search_by_name, mock_search):
    """retrieve uses name_match when routine name in query."""
    mock_embed.return_value = [0.1] * 1536
    mock_search_by_name.return_value = [
        {"id": "n1", "score": 0.95, "text": "t", "metadata": {"unit_name": "DGESV"}},
    ]
    mock_search.return_value = [
        {"id": "v1", "score": 0.8, "text": "t", "metadata": {}},
    ]

    result = await retrieve("What is DGESV?", top_k=5)
    assert result["retrieval_strategy"] == "name_match"
    assert len(result["chunks"]) >= 1
    assert any(c.get("_match_type") == "name" for c in result["chunks"])
    mock_search_by_name.assert_called()


@patch("app.services.retrieval.async_search", new_callable=AsyncMock)
@patch("app.services.retrieval.async_search_by_name", new_callable=AsyncMock)
@patch("app.services.retrieval.embed_query", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_retrieve_conceptual_uses_vector(mock_embed, mock_search_by_name, mock_search):
    """retrieve uses vector search for conceptual queries (no expansion)."""
    mock_embed.return_value = [0.1] * 1536
    mock_search_by_name.return_value = []
    mock_search.return_value = [
        {"id": "v1", "score": 0.9, "text": "t", "metadata": {}},
    ]

    result = await retrieve("How does LU decomposition work?", top_k=5)
    assert result["retrieval_strategy"] == "vector"
    assert result["expanded_names"] == []
    assert len(result["chunks"]) >= 1
    mock_search.assert_called()


@patch("app.services.retrieval.async_search", new_callable=AsyncMock)
@patch("app.services.retrieval.async_search_by_name", new_callable=AsyncMock)
@patch("app.services.retrieval.embed_query", new_callable=AsyncMock)
@patch("app.services.retrieval.get_async_openai_client")
@pytest.mark.asyncio
async def test_retrieve_expansion_exception_returns_empty(mock_client_fn, mock_embed, mock_search_by_name, mock_search):
    """_expand_query returns [] on exception, retrieval continues with vector."""
    mock_embed.return_value = [0.1] * 1536
    mock_client_fn.return_value = AsyncMock()
    mock_client_fn.return_value.chat.completions.create.side_effect = Exception("API error")
    mock_search_by_name.return_value = []
    mock_search.return_value = [
        {"id": "v1", "score": 0.8, "text": "t", "metadata": {}},
    ]

    result = await retrieve("How does LU work?", top_k=5)
    assert result["retrieval_strategy"] == "vector"
    assert len(result["chunks"]) >= 1


@patch("app.services.retrieval.async_search", new_callable=AsyncMock)
@patch("app.services.retrieval.async_search_by_name", new_callable=AsyncMock)
@patch("app.services.retrieval.embed_query", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_retrieve_call_graph_follow(mock_embed, mock_search_by_name, mock_search):
    """retrieve follows call graph from name-matched results."""
    mock_embed.return_value = [0.1] * 1536

    # search_by_name is called for the initial name match AND for call-graph lookups
    async def name_search_side_effect(embedding, name, top_k=1):
        results = {
            "DGESV": [{"id": "n1", "score": 0.95, "text": "t", "metadata": {"unit_name": "DGESV", "called_routines": ["DGETRF", "DGETRS"]}}],
            "DGETRF": [{"id": "c1", "score": 0.9, "text": "t", "metadata": {"unit_name": "DGETRF"}}],
            "DGETRS": [{"id": "c2", "score": 0.85, "text": "t", "metadata": {"unit_name": "DGETRS"}}],
        }
        return results.get(name, [])

    mock_search_by_name.side_effect = name_search_side_effect
    mock_search.return_value = []

    result = await retrieve("What is DGESV?", top_k=5)
    assert any(c.get("_match_type") == "call_graph" for c in result["chunks"])
    assert mock_search_by_name.call_count >= 2


@patch("app.services.retrieval.async_search", new_callable=AsyncMock)
@patch("app.services.retrieval.async_search_by_name", new_callable=AsyncMock)
@patch("app.services.retrieval.embed_query", new_callable=AsyncMock)
@patch("app.services.retrieval._expand_query", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_retrieve_vector_merge_dedup(mock_expand, mock_embed, mock_search_by_name, mock_search):
    """retrieve merges vector results and deduplicates by id."""
    mock_embed.return_value = [0.1] * 1536
    mock_expand.return_value = []
    mock_search_by_name.return_value = []
    mock_search.return_value = [
        {"id": "v1", "score": 0.9, "text": "t1", "metadata": {}},
        {"id": "v2", "score": 0.8, "text": "t2", "metadata": {}},
    ]

    result = await retrieve("conceptual query", top_k=5)
    assert result["retrieval_strategy"] == "vector"
    assert len(result["chunks"]) == 2
    ids = [c["id"] for c in result["chunks"]]
    assert len(ids) == len(set(ids))
