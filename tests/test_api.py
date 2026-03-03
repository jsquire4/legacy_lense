"""Smoke tests for FastAPI endpoints."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.eval_data import EVAL_QUERIES, E2E_EVAL_QUERIES

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "LegacyLens" in response.text


@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_endpoint(mock_retrieve, mock_generate):
    mock_retrieve.return_value = {
        "chunks": [
            {"id": "abc123", "text": "test", "score": 0.9, "metadata": {"file_path": "test.f"}, "_match_type": "vector"}
        ],
        "expanded_names": [],
        "retrieval_strategy": "vector",
    }
    mock_generate.return_value = {
        "answer": "Test answer",
        "citations": ["test.f:1-10"],
        "model": "gpt-4o-mini",
        "token_usage": {},
    }

    response = client.post("/api/query", json={"query": "What is DGESV?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "latency_ms" in data


@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_endpoint_cache_hit(mock_retrieve, mock_generate):
    """Second identical query returns cached response; retrieve/generate called once."""
    mock_retrieve.return_value = {
        "chunks": [
            {"id": "c1", "text": "cached", "score": 0.9, "metadata": {"file_path": "x.f"}, "_match_type": "vector"}
        ],
        "expanded_names": [],
        "retrieval_strategy": "vector",
    }
    mock_generate.return_value = {
        "answer": "Cached answer",
        "citations": ["x.f:1-5"],
        "model": "gpt-4o-mini",
        "token_usage": {},
    }

    payload = {"query": "Cache me please", "top_k": 8}
    r1 = client.post("/api/query", json=payload)
    r2 = client.post("/api/query", json=payload)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["answer"] == r2.json()["answer"]
    mock_retrieve.assert_called_once()
    mock_generate.assert_called_once()


@patch("app.main._CACHE_TTL", 0)
@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_endpoint_cache_ttl_expiry(mock_retrieve, mock_generate):
    """Cache entry expires after TTL; next request fetches fresh data (covers del path)."""
    mock_retrieve.return_value = {
        "chunks": [{"id": "x", "text": "t", "score": 0.9, "metadata": {}, "_match_type": "vector"}],
        "expanded_names": [],
        "retrieval_strategy": "vector",
    }
    mock_generate.return_value = {
        "answer": "Answer",
        "citations": [],
        "model": "gpt-4o-mini",
        "token_usage": {},
    }

    import app.main as main_mod

    main_mod._RESPONSE_CACHE.clear()
    try:
        payload = {"query": "TTL expiry test query"}
        r1 = client.post("/api/query", json=payload)
        assert r1.status_code == 200
        assert mock_retrieve.call_count == 1

        r2 = client.post("/api/query", json=payload)
        assert r2.status_code == 200
        assert mock_retrieve.call_count == 2
    finally:
        main_mod._RESPONSE_CACHE.clear()


@patch("app.main._CACHE_MAX", 3)
@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_endpoint_cache_eviction(mock_retrieve, mock_generate):
    """Cache evicts oldest when exceeding _CACHE_MAX."""
    mock_retrieve.return_value = {
        "chunks": [{"id": "x", "text": "t", "score": 0.9, "metadata": {}, "_match_type": "vector"}],
        "expanded_names": [],
        "retrieval_strategy": "vector",
    }
    mock_generate.return_value = {
        "answer": "Answer",
        "citations": [],
        "model": "gpt-4o-mini",
        "token_usage": {},
    }

    import app.main as main_mod

    main_mod._RESPONSE_CACHE.clear()
    try:
        for i in range(5):
            client.post("/api/query", json={"query": f"Distinct query {i} unique"})
        assert mock_retrieve.call_count == 5
        assert len(main_mod._RESPONSE_CACHE) <= 3
    finally:
        main_mod._RESPONSE_CACHE.clear()


@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_capability_endpoint(mock_retrieve, mock_generate):
    mock_retrieve.return_value = {
        "chunks": [{"id": "abc123", "text": "test", "score": 0.9, "metadata": {}, "_match_type": "vector"}],
        "expanded_names": [],
        "retrieval_strategy": "vector",
    }
    mock_generate.return_value = {
        "answer": "Explained",
        "citations": [],
        "model": "gpt-4o-mini",
        "token_usage": {},
    }

    response = client.post(
        "/api/capabilities/explain_code",
        json={"query": "Explain DGESV"},
    )
    assert response.status_code == 200


def test_unknown_capability():
    response = client.post(
        "/api/capabilities/nonexistent",
        json={"query": "test"},
    )
    assert response.status_code == 404


def test_unknown_capability_stream():
    """Capability stream endpoint returns 404 for unknown capability."""
    response = client.post(
        "/api/capabilities/nonexistent/stream",
        json={"query": "test"},
    )
    assert response.status_code == 404


def test_query_request_validation_empty_query():
    """QueryRequest rejects empty query."""
    response = client.post("/api/query", json={"query": ""})
    assert response.status_code == 422


def test_query_request_validation_top_k_bounds():
    """QueryRequest enforces top_k between 1 and 20."""
    response_lo = client.post("/api/query", json={"query": "test", "top_k": 0})
    assert response_lo.status_code == 422
    response_hi = client.post("/api/query", json={"query": "test", "top_k": 21})
    assert response_hi.status_code == 422


@patch("app.main.setup_logging")
def test_lifespan_calls_setup_logging(mock_setup_logging):
    """Lifespan context calls setup_logging on startup."""
    with TestClient(app) as c:
        c.get("/health")
    mock_setup_logging.assert_called()


def _parse_sse_events(text: str) -> list[dict]:
    """Parse SSE text into list of {event, data} dicts."""
    import json
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = data = None
        for line in block.strip().split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event and data is not None:
            events.append({"event": event, "data": data})
    return events


@patch("app.main.generate_answer_stream")
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_stream_endpoint(mock_retrieve, mock_stream):
    mock_retrieve.return_value = {
        "chunks": [
            {"id": "abc123", "text": "test", "score": 0.9, "metadata": {"file_path": "test.f"}, "_match_type": "vector"}
        ],
        "expanded_names": [],
        "retrieval_strategy": "vector",
    }

    async def async_gen(*args, **kwargs):
        yield {"type": "token", "token": "Hello"}
        yield {"type": "token", "token": " world"}
        yield {"type": "done", "citations": ["test.f:1-10"], "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    mock_stream.side_effect = async_gen

    response = client.post("/api/query/stream", json={"query": "What is DGESV?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    event_types = [e["event"] for e in events]
    assert "retrieval" in event_types
    assert "token" in event_types
    assert "done" in event_types


@patch("app.main.generate_answer_stream")
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_stream_skips_unknown_event_types(mock_retrieve, mock_stream):
    """Stream generator skips events with type other than token/done (branch coverage)."""
    mock_retrieve.return_value = {
        "chunks": [{"id": "x", "text": "t", "score": 0.9, "metadata": {}, "_match_type": "vector"}],
        "expanded_names": [],
        "retrieval_strategy": "vector",
    }

    async def async_gen(*args, **kwargs):
        yield {"type": "unknown"}  # skipped
        yield {"type": "token", "token": "Hi"}
        yield {"type": "done", "citations": [], "token_usage": {}}

    mock_stream.side_effect = async_gen

    response = client.post("/api/query/stream", json={"query": "test"})
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert events[-1]["event"] == "done"


@patch("app.main.generate_answer_stream")
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_capability_stream_endpoint(mock_retrieve, mock_stream):
    mock_retrieve.return_value = {
        "chunks": [{"id": "abc123", "text": "test", "score": 0.9, "metadata": {}, "_match_type": "vector"}],
        "expanded_names": [],
        "retrieval_strategy": "vector",
    }

    async def async_gen(*args, **kwargs):
        yield {"type": "token", "token": "Explained"}
        yield {"type": "done", "citations": [], "token_usage": {}}

    mock_stream.side_effect = async_gen

    response = client.post("/api/capabilities/explain_code/stream", json={"query": "Explain DGESV"})
    assert response.status_code == 200

    events = _parse_sse_events(response.text)
    assert events[-1]["event"] == "done"


@patch("app.main.retrieve", new_callable=AsyncMock)
def test_eval_stream_returns_sse(mock_retrieve):
    mock_retrieve.return_value = {
        "chunks": [
            {"id": "x1", "text": "t", "score": 0.9,
             "metadata": {"file_path": "dgesv.f"}, "_match_type": "name"}
        ],
        "expanded_names": [],
        "retrieval_strategy": "name_match",
    }

    response = client.get("/api/eval/stream")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


@patch("app.main.retrieve", new_callable=AsyncMock)
def test_eval_stream_chunks_without_file_path(mock_retrieve):
    """Eval stream handles chunks with no file_path in metadata (branch coverage)."""
    mock_retrieve.return_value = {
        "chunks": [
            {"id": "x1", "text": "t", "score": 0.9, "metadata": {}, "_match_type": "name"},
            {"id": "x2", "text": "t", "score": 0.8, "metadata": {"file_path": "dgesv.f"}, "_match_type": "vector"},
        ],
        "expanded_names": [],
        "retrieval_strategy": "vector",
    }

    response = client.get("/api/eval/stream")
    events = _parse_sse_events(response.text)
    progress_events = [e for e in events if e["event"] == "progress"]
    assert len(progress_events) == len(EVAL_QUERIES)
    first = progress_events[0]["data"]
    assert "retrieved_files" in first


@patch("app.main.retrieve", new_callable=AsyncMock)
def test_eval_stream_deduplicates_file_paths(mock_retrieve):
    """Eval stream deduplicates retrieved_files when multiple chunks share file_path (branch coverage)."""
    mock_retrieve.return_value = {
        "chunks": [
            {"id": "x1", "text": "t", "score": 0.9, "metadata": {"file_path": "/path/dgesv.f"}, "_match_type": "name"},
            {"id": "x2", "text": "t2", "score": 0.85, "metadata": {"file_path": "/other/dgesv.f"}, "_match_type": "vector"},
        ],
        "expanded_names": [],
        "retrieval_strategy": "vector",
    }

    response = client.get("/api/eval/stream")
    events = _parse_sse_events(response.text)
    first_progress = [e for e in events if e["event"] == "progress"][0]["data"]
    assert first_progress["retrieved_files"].count("dgesv.f") == 1


@patch("app.main.retrieve", new_callable=AsyncMock)
def test_eval_stream_emits_progress_events(mock_retrieve):
    mock_retrieve.return_value = {
        "chunks": [
            {"id": "x1", "text": "t", "score": 0.9,
             "metadata": {"file_path": "dgesv.f"}, "_match_type": "name"}
        ],
        "expanded_names": [],
        "retrieval_strategy": "name_match",
    }

    response = client.get("/api/eval/stream")
    events = _parse_sse_events(response.text)

    progress_events = [e for e in events if e["event"] == "progress"]
    assert len(progress_events) == len(EVAL_QUERIES)

    first = progress_events[0]["data"]
    assert "query" in first
    assert "capability" in first
    assert "recall_at_5" in first
    assert "latency_ms" in first
    assert "retrieved_files" in first
    assert "expected_files" in first
    assert first["index"] == 0


@patch("app.main.retrieve", new_callable=AsyncMock)
def test_eval_stream_summary_is_last_event(mock_retrieve):
    mock_retrieve.return_value = {
        "chunks": [
            {"id": "x1", "text": "t", "score": 0.9,
             "metadata": {"file_path": "dgesv.f"}, "_match_type": "name"}
        ],
        "expanded_names": [],
        "retrieval_strategy": "name_match",
    }

    response = client.get("/api/eval/stream")
    events = _parse_sse_events(response.text)

    assert events[-1]["event"] == "summary"
    summary = events[-1]["data"]
    assert "avg_recall_at_5" in summary
    assert "avg_latency_ms" in summary
    assert summary["total_queries"] == len(EVAL_QUERIES)
    assert 0.0 <= summary["avg_recall_at_5"] <= 1.0


@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_e2e_eval_stream_endpoint(mock_retrieve, mock_generate):
    mock_retrieve.return_value = {
        "chunks": [
            {"id": "x1", "text": "t", "score": 0.9,
             "metadata": {"file_path": "dgesv.f"}, "_match_type": "name"}
        ],
        "expanded_names": [],
        "retrieval_strategy": "name_match",
    }
    mock_generate.return_value = {
        "answer": (
            "DGESV solves a linear system of equations using LU factorization with partial pivoting. "
            "It calls DGETRF and DGETRS. The routine performs matrix multiplication via DGEMM and "
            "triangular solve with DTRSM. DGEQRF handles QR. The norm of the matrix is computed. "
            "Singular value decomposition is used. Cholesky symmetric factorization via DPOTRF. "
            "Least square problems solved by DGELS. Error check and workspace query with LWORK. "
            "Loop and block optimizations improve performance. Each routine validates dimension and LDA. "
            "INFO parameter and N must be positive."
        ),
        "citations": ["dgesv.f:1-50"],
        "model": "gpt-4o-mini",
        "token_usage": {},
    }

    response = client.get("/api/eval/e2e/stream")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    progress_events = [e for e in events if e["event"] == "progress"]
    assert len(progress_events) == len(E2E_EVAL_QUERIES)

    first = progress_events[0]["data"]
    assert "passed" in first
    assert "capability" in first
    assert "checks" in first
    assert "latency_ms" in first
    assert "answer_preview" in first
    assert "citations" in first
    assert isinstance(first["citations"], list)

    assert events[-1]["event"] == "summary"
    summary = events[-1]["data"]
    assert "pass_rate" in summary
    assert "passed" in summary
    assert "failed" in summary
    assert summary["total_queries"] == len(E2E_EVAL_QUERIES)


@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_e2e_eval_stream_includes_failed_checks(mock_retrieve, mock_generate):
    """E2E eval stream includes progress events when checks fail (branch coverage)."""
    mock_retrieve.return_value = {
        "chunks": [
            {"id": "x1", "text": "t", "score": 0.9,
             "metadata": {"file_path": "dgesv.f"}, "_match_type": "name"}
        ],
        "expanded_names": [],
        "retrieval_strategy": "name_match",
    }
    # Answer that fails: too short, no expected keywords
    mock_generate.return_value = {
        "answer": "Short.",
        "citations": [],
        "model": "gpt-4o-mini",
        "token_usage": {},
    }

    response = client.get("/api/eval/e2e/stream")
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    progress_events = [e for e in events if e["event"] == "progress"]
    assert len(progress_events) >= 1
    # At least one should have passed=False due to our mock
    failed = [p for p in progress_events if not p["data"].get("passed", True)]
    assert len(failed) >= 1


# --- Model endpoint tests ---

def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    models = data["models"]
    assert len(models) >= 9
    names = [m["name"] for m in models]
    assert "gpt-4o-mini" in names
    assert "gpt-4o" in names
    # Check default flag
    defaults = [m for m in models if m.get("default")]
    assert len(defaults) == 1
    # Check pricing fields
    for m in models:
        assert "input_cost_per_1m" in m
        assert "output_cost_per_1m" in m


# --- Trial CRUD endpoint tests ---

@patch("app.main.save_trial")
def test_create_trial_endpoint(mock_save):
    mock_save.return_value = 1
    response = client.post("/api/trials", json={
        "model": "gpt-4o-mini",
        "eval_type": "retrieval",
        "avg_recall_at_5": 0.85,
        "total_queries": 10,
    })
    assert response.status_code == 200
    assert response.json() == {"id": 1}
    mock_save.assert_called_once()


@patch("app.main.list_trials")
def test_list_trials_endpoint(mock_list):
    mock_list.return_value = [
        {"id": 1, "model": "gpt-4o-mini", "eval_type": "retrieval"},
    ]
    response = client.get("/api/trials")
    assert response.status_code == 200
    assert len(response.json()["trials"]) == 1


@patch("app.main.delete_trial")
def test_delete_trial_endpoint(mock_delete):
    mock_delete.return_value = True
    response = client.delete("/api/trials/1")
    assert response.status_code == 200
    assert response.json() == {"deleted": True}


@patch("app.main.delete_trial")
def test_delete_trial_not_found(mock_delete):
    mock_delete.return_value = False
    response = client.delete("/api/trials/999")
    assert response.status_code == 404


# --- Model param on query/eval endpoints ---

@patch("app.main.generate_answer", new_callable=AsyncMock)
@patch("app.main.retrieve", new_callable=AsyncMock)
def test_query_endpoint_with_model(mock_retrieve, mock_generate):
    """Query endpoint passes model param through."""
    mock_retrieve.return_value = {
        "chunks": [{"id": "x", "text": "t", "score": 0.9, "metadata": {"file_path": "t.f"}, "_match_type": "vector"}],
        "expanded_names": [],
        "retrieval_strategy": "vector",
    }
    mock_generate.return_value = {
        "answer": "Answer", "citations": [], "model": "gpt-4o", "token_usage": {},
    }

    import app.main as main_mod
    main_mod._RESPONSE_CACHE.clear()
    try:
        response = client.post("/api/query", json={"query": "test model param", "model": "gpt-4o"})
        assert response.status_code == 200
        mock_retrieve.assert_called_once()
        _, kwargs = mock_retrieve.call_args
        assert kwargs.get("model") == "gpt-4o" or mock_retrieve.call_args[1].get("model") == "gpt-4o"
    finally:
        main_mod._RESPONSE_CACHE.clear()


@patch("app.main.retrieve", new_callable=AsyncMock)
def test_eval_stream_with_model_param(mock_retrieve):
    """Eval stream endpoint accepts model query param."""
    mock_retrieve.return_value = {
        "chunks": [{"id": "x", "text": "t", "score": 0.9, "metadata": {"file_path": "dgesv.f"}, "_match_type": "name"}],
        "expanded_names": [],
        "retrieval_strategy": "name_match",
    }
    response = client.get("/api/eval/stream?model=gpt-4o")
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert any(e["event"] == "summary" for e in events)


@patch("app.main._build_response", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_warm_cache_success(mock_build):
    """_warm_cache calls _build_response for each warmup query * model."""
    mock_build.return_value = MagicMock()

    from app.main import _warm_cache, _WARMUP_QUERIES, _WARMUP_MODELS
    await _warm_cache()
    assert mock_build.call_count == len(_WARMUP_QUERIES) * len(_WARMUP_MODELS)


@patch("app.main._build_response", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_warm_cache_handles_errors(mock_build):
    """_warm_cache continues when a query fails."""
    mock_build.side_effect = Exception("API down")

    from app.main import _warm_cache, _WARMUP_QUERIES, _WARMUP_MODELS
    await _warm_cache()  # should not raise
    assert mock_build.call_count == len(_WARMUP_QUERIES) * len(_WARMUP_MODELS)
