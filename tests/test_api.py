"""Smoke tests for FastAPI endpoints."""

from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "LegacyLens" in response.text


@patch("app.main.generate_answer")
@patch("app.main.retrieve")
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


@patch("app.main.generate_answer")
@patch("app.main.retrieve")
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
@patch("app.main.retrieve")
def test_query_stream_endpoint(mock_retrieve, mock_stream):
    mock_retrieve.return_value = {
        "chunks": [
            {"id": "abc123", "text": "test", "score": 0.9, "metadata": {"file_path": "test.f"}, "_match_type": "vector"}
        ],
        "expanded_names": [],
        "retrieval_strategy": "vector",
    }
    mock_stream.return_value = iter([
        {"type": "token", "token": "Hello"},
        {"type": "token", "token": " world"},
        {"type": "done", "citations": ["test.f:1-10"], "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    ])

    response = client.post("/api/query/stream", json={"query": "What is DGESV?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    event_types = [e["event"] for e in events]
    assert "retrieval" in event_types
    assert "token" in event_types
    assert "done" in event_types


@patch("app.main.generate_answer_stream")
@patch("app.main.retrieve")
def test_capability_stream_endpoint(mock_retrieve, mock_stream):
    mock_retrieve.return_value = {
        "chunks": [{"id": "abc123", "text": "test", "score": 0.9, "metadata": {}, "_match_type": "vector"}],
        "expanded_names": [],
        "retrieval_strategy": "vector",
    }
    mock_stream.return_value = iter([
        {"type": "token", "token": "Explained"},
        {"type": "done", "citations": [], "token_usage": {}},
    ])

    response = client.post("/api/capabilities/explain_code/stream", json={"query": "Explain DGESV"})
    assert response.status_code == 200

    events = _parse_sse_events(response.text)
    assert events[-1]["event"] == "done"
