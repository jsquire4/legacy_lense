"""Tests for the retrieval service (mocked, no API/Qdrant calls)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.retrieval import _extract_routine_name, _expand_query, retrieve


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


@patch("app.services.retrieval.get_async_openai_client")
@patch("app.services.retrieval.get_settings")
@pytest.mark.asyncio
async def test_expand_query_success(mock_settings, mock_client_fn):
    """_expand_query returns LAPACK names from LLM response."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_msg = MagicMock()
    mock_msg.content = "DGESV DGETRF DGETRS DGESVX"
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = mock_msg

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_client_fn.return_value = mock_client

    result = await _expand_query("How does LU decomposition work?")
    assert "DGESV" in result
    assert "DGETRF" in result
    mock_client.chat.completions.create.assert_called_once()


@patch("app.services.retrieval.get_async_openai_client")
@patch("app.services.retrieval.get_settings")
@pytest.mark.asyncio
async def test_expand_query_exception_returns_empty(mock_settings, mock_client_fn):
    """_expand_query returns [] on exception."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
    mock_client_fn.return_value = mock_client

    result = await _expand_query("How does SVD work?")
    assert result == []


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
@patch("app.services.retrieval._expand_query", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_retrieve_conceptual_uses_expansion(mock_expand, mock_embed, mock_search_by_name, mock_search):
    """retrieve uses LLM expansion for conceptual queries without routine names."""
    mock_embed.return_value = [0.1] * 1536
    mock_expand.return_value = ["DGETRF", "DGESV"]
    mock_search_by_name.return_value = [
        {"id": "e1", "score": 0.9, "text": "t", "metadata": {"unit_name": "DGETRF"}},
    ]
    mock_search.return_value = [
        {"id": "v1", "score": 0.8, "text": "t", "metadata": {}},
    ]

    result = await retrieve("How does LU decomposition work?", top_k=5)
    assert result["retrieval_strategy"] == "expansion"
    assert result["expanded_names"] == ["DGETRF", "DGESV"]
    assert any(c.get("_match_type") == "expansion" for c in result["chunks"])
    mock_expand.assert_called_once()
    mock_search_by_name.assert_called()


@patch("app.services.retrieval.async_search", new_callable=AsyncMock)
@patch("app.services.retrieval.async_search_by_name", new_callable=AsyncMock)
@patch("app.services.retrieval.embed_query", new_callable=AsyncMock)
@patch("app.services.retrieval._expand_query", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_retrieve_conceptual_falls_back_to_vector(mock_expand, mock_embed, mock_search_by_name, mock_search):
    """retrieve falls back to vector-only when expansion returns no names."""
    mock_embed.return_value = [0.1] * 1536
    mock_expand.return_value = []
    mock_search_by_name.return_value = []
    mock_search.return_value = [
        {"id": "v1", "score": 0.9, "text": "t", "metadata": {}},
    ]

    result = await retrieve("How does matrix math work?", top_k=5)
    assert result["retrieval_strategy"] == "vector"
    assert result["expanded_names"] == []
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


@patch("app.services.retrieval.async_search", new_callable=AsyncMock)
@patch("app.services.retrieval.async_search_by_name", new_callable=AsyncMock)
@patch("app.services.retrieval.embed_query", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_retrieve_name_match_skips_duplicate_ids(mock_embed, mock_search_by_name, mock_search):
    """retrieve skips name-match hits with duplicate ids (branch coverage)."""
    mock_embed.return_value = [0.1] * 1536
    mock_search_by_name.return_value = [
        {"id": "n1", "score": 0.95, "text": "t", "metadata": {"unit_name": "DGESV"}},
        {"id": "n1", "score": 0.9, "text": "t2", "metadata": {"unit_name": "DGESV"}},
    ]
    mock_search.return_value = []

    result = await retrieve("What is DGESV?", top_k=5)
    assert result["retrieval_strategy"] == "name_match"
    assert len(result["chunks"]) == 1


@patch("app.services.retrieval.async_search", new_callable=AsyncMock)
@patch("app.services.retrieval.async_search_by_name", new_callable=AsyncMock)
@patch("app.services.retrieval.embed_query", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_retrieve_skips_vector_duplicates(mock_embed, mock_search_by_name, mock_search):
    """retrieve skips vector results already in seen_ids (branch coverage)."""
    mock_embed.return_value = [0.1] * 1536
    mock_search_by_name.return_value = [
        {"id": "shared", "score": 0.95, "text": "t", "metadata": {"unit_name": "DGESV"}},
    ]
    mock_search.return_value = [
        {"id": "shared", "score": 0.8, "text": "t2", "metadata": {}},
        {"id": "v2", "score": 0.7, "text": "t3", "metadata": {}},
    ]

    result = await retrieve("What is DGESV?", top_k=5)
    ids = [c["id"] for c in result["chunks"]]
    assert ids.count("shared") == 1


@patch("app.services.retrieval.async_search", new_callable=AsyncMock)
@patch("app.services.retrieval.async_search_by_name", new_callable=AsyncMock)
@patch("app.services.retrieval.embed_query", new_callable=AsyncMock)
@patch("app.services.retrieval._expand_query", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_retrieve_passes_model_to_expand_query(mock_expand, mock_embed, mock_search_by_name, mock_search):
    """retrieve passes model override to _expand_query."""
    mock_embed.return_value = [0.1] * 1536
    mock_expand.return_value = []
    mock_search_by_name.return_value = []
    mock_search.return_value = [
        {"id": "v1", "score": 0.9, "text": "t", "metadata": {}},
    ]

    await retrieve("How does SVD work?", top_k=5, model="gpt-4o")
    mock_expand.assert_called_once_with("How does SVD work?", model="gpt-4o")


@patch("app.services.retrieval.get_async_openai_client")
@patch("app.services.retrieval.get_settings")
@pytest.mark.asyncio
async def test_expand_query_uses_model_override(mock_settings, mock_client_fn):
    """_expand_query uses provided model instead of settings.CHAT_MODEL."""
    settings = MagicMock()
    settings.CHAT_MODEL = "gpt-4o-mini"
    mock_settings.return_value = settings

    mock_msg = MagicMock()
    mock_msg.content = "DGESVD DGESDD"
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = mock_msg

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_client_fn.return_value = mock_client

    await _expand_query("How does SVD work?", model="gpt-4o")
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs.get("model") == "gpt-4o" or call_kwargs[1].get("model") == "gpt-4o"


@patch("app.services.retrieval.async_search", new_callable=AsyncMock)
@patch("app.services.retrieval.async_search_by_name", new_callable=AsyncMock)
@patch("app.services.retrieval.embed_query", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_retrieve_stops_merge_when_top_k_full(mock_embed, mock_search_by_name, mock_search):
    """retrieve stops adding vector results when merged reaches top_k (branch coverage)."""
    mock_embed.return_value = [0.1] * 1536
    mock_search_by_name.return_value = [
        {"id": "n1", "score": 0.99, "text": "t", "metadata": {"unit_name": "DGESV"}},
        {"id": "n2", "score": 0.98, "text": "t", "metadata": {"unit_name": "DGESV"}},
        {"id": "n3", "score": 0.97, "text": "t", "metadata": {"unit_name": "DGESV"}},
    ]
    mock_search.return_value = [
        {"id": "v1", "score": 0.5, "text": "t", "metadata": {}},
        {"id": "v2", "score": 0.4, "text": "t", "metadata": {}},
    ]

    result = await retrieve("What is DGESV?", top_k=3)
    assert len(result["chunks"]) == 3
    assert "v1" not in [c["id"] for c in result["chunks"]]
