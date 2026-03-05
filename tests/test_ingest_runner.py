"""Tests for the ingestion benchmarking SSE stream generator."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.services.ingest_runner import ingest_stream_generator, _find_fortran_files
from tests.helpers import collect_sse_events


def _patch_no_fixture():
    """Patch DEFAULT_FIXTURE_PATH.exists() to return False, forcing source-parsing path."""
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = False
    return patch("app.services.ingest_runner.DEFAULT_FIXTURE_PATH", mock_path)


@pytest.mark.asyncio
async def test_unknown_model_emits_error():
    events = await collect_sse_events(ingest_stream_generator("nonexistent-model"))
    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert "Unknown embedding model" in events[0]["data"]["message"]


def test_find_fortran_files_finds_files():
    """_find_fortran_files discovers .f files in SRC subdir (lines 31-40)."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "SRC"
        src.mkdir()
        (src / "dgesv.f").write_text("      SUBROUTINE DGESV\n      END\n")
        (src / "dgetrf.f").write_text("      SUBROUTINE DGETRF\n      END\n")
        files = _find_fortran_files(Path(tmp), [".f"], ["SRC"])
        assert len(files) == 2
        names = {f.name for f in files}
        assert "dgesv.f" in names
        assert "dgetrf.f" in names


def test_find_fortran_files_skips_missing_subdir():
    """_find_fortran_files skips subdirs that don't exist (line 35)."""
    with tempfile.TemporaryDirectory() as tmp:
        # No SRC dir - subdir_path.exists() is False, we continue
        files = _find_fortran_files(Path(tmp), [".f"], ["SRC"])
        assert len(files) == 0


@pytest.mark.asyncio
@patch("app.services.ingest_runner._ingest_lock")
async def test_ingest_lock_held_emits_error(mock_lock):
    """Ingestion emits error when lock is already held (lines 31-40)."""
    mock_lock.locked.return_value = True
    events = await collect_sse_events(
        ingest_stream_generator("text-embedding-3-small")
    )
    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert "already running" in events[0]["data"]["message"]


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
    with patch.object(Path, "exists", return_value=True), _patch_no_fixture():
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
    assert parse_events[0]["data"]["source"] == "files"
    assert parse_events[0]["data"]["files"] == 1
    assert parse_events[0]["data"]["files_parsed"] == 1
    assert parse_events[0]["data"]["parse_errors"] == 0
    assert parse_events[0]["data"]["units"] == 1
    assert parse_events[0]["data"]["chunks"] == 1
    assert parse_events[0]["data"]["coverage_pct"] == 100.0

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
    assert summary["files_parsed"] == 1
    assert summary["parse_errors"] == 0
    assert summary["coverage_pct"] == 100.0
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
    with patch.object(Path, "exists", return_value=False), _patch_no_fixture():
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
    with patch.object(Path, "exists", return_value=True), _patch_no_fixture():
        mock_find.return_value = []
        events = await collect_sse_events(
            ingest_stream_generator("text-embedding-3-small")
        )

    assert len(events) == 1
    assert events[0]["event"] == "error"
    assert "No Fortran files" in events[0]["data"]["message"]


# --- Audit issue #21: Ingest error handling ---

@pytest.mark.asyncio
@patch("app.services.ingest_runner.upsert_chunks")
@patch("app.services.ingest_runner.embed_texts")
@patch("app.services.ingest_runner.ensure_collection")
@patch("app.services.ingest_runner.delete_collection")
@patch("app.services.ingest_runner.chunk_units")
@patch("app.services.ingest_runner.parse_file")
@patch("app.services.ingest_runner._find_fortran_files")
@patch("app.services.ingest_runner.get_settings")
async def test_ingest_rate_limit_retry_emits_progress(
    mock_settings, mock_find, mock_parse, mock_chunk,
    mock_delete_coll, mock_ensure_coll, mock_embed, mock_upsert,
):
    """Ingestion emits rate_limited progress when embed_texts raises 429 then succeeds (lines 130-139)."""
    settings = MagicMock()
    settings.DATA_DIR = "/tmp/fake_data"
    mock_settings.return_value = settings

    from pathlib import Path
    with patch.object(Path, "exists", return_value=True), _patch_no_fixture():
        mock_find.return_value = [Path("/tmp/fake_data/SRC/dgesv.f")]

        mock_unit = MagicMock()
        mock_unit.name = "DGESV"
        mock_unit.called_routines = []
        mock_parse.return_value = [mock_unit]

        mock_chunk_obj = MagicMock()
        mock_chunk_obj.text = "test"
        mock_chunk.return_value = [mock_chunk_obj]

        call_count = 0
        def embed_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return [[0.1] * 1536]
        mock_embed.side_effect = embed_side_effect

        events = await collect_sse_events(
            ingest_stream_generator("text-embedding-3-small")
        )

    rate_limited = [e for e in events if e["event"] == "progress" and e["data"].get("phase") == "rate_limited"]
    assert len(rate_limited) >= 1
    assert "summary" in [e["event"] for e in events]


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
    with patch.object(Path, "exists", return_value=True), _patch_no_fixture():
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
    with patch.object(Path, "exists", return_value=True), _patch_no_fixture():
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
