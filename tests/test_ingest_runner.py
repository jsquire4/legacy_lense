"""Tests for the ingestion benchmarking SSE stream generator."""

from unittest.mock import patch, MagicMock

import pytest

from app.services.ingest_runner import ingest_stream_generator
from tests.helpers import collect_sse_events


@pytest.mark.asyncio
async def test_unknown_model_emits_error():
    events = await collect_sse_events(ingest_stream_generator("nonexistent-model"))
    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert "Unknown embedding model" in events[0]["data"]["message"]


@pytest.mark.asyncio
@patch("app.services.ingest_runner.upsert_chunks")
@patch("app.services.ingest_runner.embed_texts")
@patch("app.services.ingest_runner.ensure_collection")
@patch("app.services.ingest_runner.delete_collection")
@patch("app.services.ingest_runner.chunk_units")
@patch("app.services.ingest_runner.parse_file")
@patch("app.services.ingest_runner._find_fortran_files")
@patch("app.services.ingest_runner.get_settings")
async def test_successful_ingestion_emits_correct_events(
    mock_settings, mock_find, mock_parse, mock_chunk,
    mock_delete_coll, mock_ensure_coll, mock_embed, mock_upsert,
):
    # Setup
    settings = MagicMock()
    settings.DATA_DIR = "/tmp/fake_data"
    mock_settings.return_value = settings

    from pathlib import Path
    with patch.object(Path, "exists", return_value=True):
        mock_find.return_value = [Path("/tmp/fake_data/SRC/dgesv.f")]

        mock_unit = MagicMock()
        mock_unit.name = "DGESV"
        mock_unit.called_routines = ["DGETRF"]
        mock_parse.return_value = [mock_unit]

        mock_chunk_obj = MagicMock()
        mock_chunk_obj.text = "test chunk text"
        mock_chunk.return_value = [mock_chunk_obj]

        mock_embed.return_value = [[0.1] * 1536]
        mock_delete_coll.return_value = True

        events = await collect_sse_events(
            ingest_stream_generator("text-embedding-3-small")
        )

    event_types = [e["event"] for e in events]
    assert "progress" in event_types
    assert "summary" in event_types

    # Check parsing progress event
    parse_events = [e for e in events if e["event"] == "progress" and e["data"].get("phase") == "parsing"]
    assert len(parse_events) == 1
    assert parse_events[0]["data"]["files"] == 1
    assert parse_events[0]["data"]["units"] == 1
    assert parse_events[0]["data"]["chunks"] == 1

    # Check embedding progress event
    embed_events = [e for e in events if e["event"] == "progress" and e["data"].get("phase") == "embedding"]
    assert len(embed_events) == 1
    assert embed_events[0]["data"]["completed"] == 1
    assert embed_events[0]["data"]["total"] == 1

    # Check summary
    summary = [e for e in events if e["event"] == "summary"][0]["data"]
    assert summary["embedding_model"] == "text-embedding-3-small"
    assert summary["dimensions"] == 1536
    assert summary["files_processed"] == 1
    assert summary["chunks_ingested"] == 1
    assert "ingestion_time_sec" in summary
    assert "chunks_per_sec" in summary

    mock_delete_coll.assert_called_once()
    mock_ensure_coll.assert_called_once()
    mock_embed.assert_called_once()
    mock_upsert.assert_called_once()


@pytest.mark.asyncio
@patch("app.services.ingest_runner._find_fortran_files")
@patch("app.services.ingest_runner.get_settings")
async def test_missing_data_dir_emits_error(mock_settings, mock_find):
    settings = MagicMock()
    settings.DATA_DIR = "/nonexistent/path"
    mock_settings.return_value = settings

    from pathlib import Path
    with patch.object(Path, "exists", return_value=False):
        events = await collect_sse_events(
            ingest_stream_generator("text-embedding-3-small")
        )

    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert "not found" in events[0]["data"]["message"]


@pytest.mark.asyncio
@patch("app.services.ingest_runner._find_fortran_files")
@patch("app.services.ingest_runner.get_settings")
async def test_no_files_found_emits_error(mock_settings, mock_find):
    settings = MagicMock()
    settings.DATA_DIR = "/tmp/fake_data"
    mock_settings.return_value = settings

    from pathlib import Path
    with patch.object(Path, "exists", return_value=True):
        mock_find.return_value = []
        events = await collect_sse_events(
            ingest_stream_generator("text-embedding-3-small")
        )

    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert "No Fortran files" in events[0]["data"]["message"]


# --- Audit issue #21: Ingest error handling ---

@pytest.mark.asyncio
@patch("app.services.ingest_runner.embed_texts")
@patch("app.services.ingest_runner.ensure_collection")
@patch("app.services.ingest_runner.delete_collection")
@patch("app.services.ingest_runner.chunk_units")
@patch("app.services.ingest_runner.parse_file")
@patch("app.services.ingest_runner._find_fortran_files")
@patch("app.services.ingest_runner.get_settings")
async def test_ingest_embed_error(
    mock_settings, mock_find, mock_parse, mock_chunk,
    mock_delete_coll, mock_ensure_coll, mock_embed,
):
    """Ingestion emits error event when embed_texts raises."""
    settings = MagicMock()
    settings.DATA_DIR = "/tmp/fake_data"
    mock_settings.return_value = settings

    from pathlib import Path
    with patch.object(Path, "exists", return_value=True):
        mock_find.return_value = [Path("/tmp/fake_data/SRC/dgesv.f")]

        mock_unit = MagicMock()
        mock_unit.name = "DGESV"
        mock_unit.called_routines = []
        mock_parse.return_value = [mock_unit]

        mock_chunk_obj = MagicMock()
        mock_chunk_obj.text = "test chunk text"
        mock_chunk.return_value = [mock_chunk_obj]

        mock_embed.side_effect = RuntimeError("API key invalid")

        events = await collect_sse_events(
            ingest_stream_generator("text-embedding-3-small")
        )

    event_types = [e["event"] for e in events]
    assert "progress" in event_types  # parsing progress emitted before failure
    assert "error" in event_types
    assert "API key invalid" in events[-1]["data"]["message"]
    assert "summary" not in event_types


@pytest.mark.asyncio
@patch("app.services.ingest_runner.chunk_units")
@patch("app.services.ingest_runner.parse_file")
@patch("app.services.ingest_runner._find_fortran_files")
@patch("app.services.ingest_runner.get_settings")
async def test_ingest_parse_error_partial_success(
    mock_settings, mock_find, mock_parse, mock_chunk,
):
    """Ingestion continues when parse_file raises on one file (partial success)."""
    settings = MagicMock()
    settings.DATA_DIR = "/tmp/fake_data"
    mock_settings.return_value = settings

    from pathlib import Path
    with patch.object(Path, "exists", return_value=True):
        file1 = Path("/tmp/fake_data/SRC/bad.f")
        file2 = Path("/tmp/fake_data/SRC/good.f")
        mock_find.return_value = [file1, file2]

        mock_unit = MagicMock()
        mock_unit.name = "GOOD"
        mock_unit.called_routines = []

        def parse_side_effect(fp):
            if fp == file1:
                raise RuntimeError("Parse error")
            return [mock_unit]

        mock_parse.side_effect = parse_side_effect
        mock_chunk.return_value = []  # No chunks since we're testing parse

        with patch("app.services.ingest_runner.delete_collection"), \
             patch("app.services.ingest_runner.ensure_collection"), \
             patch("app.services.ingest_runner.embed_texts", return_value=[]), \
             patch("app.services.ingest_runner.upsert_chunks"):
            events = await collect_sse_events(
                ingest_stream_generator("text-embedding-3-small")
            )

    event_types = [e["event"] for e in events]
    # Should still produce summary even with one file failing to parse
    assert "summary" in event_types
