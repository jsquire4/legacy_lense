"""Tests for the chunker service."""

import pytest
from app.services.parser import ParsedUnit
from app.services.chunker import chunk_units, _count_tokens


def _make_unit(source_text="X = 1", name="TEST", kind="subroutine",
               doc_comments="", file_path="test.f", called_routines=None):
    return ParsedUnit(
        name=name, kind=kind, source_text=source_text,
        doc_comments=doc_comments, file_path=file_path,
        start_line=1, end_line=10, called_routines=called_routines or [],
    )


def test_single_chunk_under_limit():
    unit = _make_unit(source_text="X = 1\nY = 2\nZ = 3")
    chunks = chunk_units([unit], max_tokens=8191)
    assert len(chunks) == 1
    assert "TEST" in chunks[0].text
    assert "test.f" in chunks[0].text
    assert chunks[0].metadata["unit_name"] == "TEST"


def test_metadata_header_present():
    unit = _make_unit(source_text="CALL DGEMM", called_routines=["DGEMM"])
    chunks = chunk_units([unit])
    assert "FILE: test.f" in chunks[0].text
    assert "ROUTINE: TEST (subroutine)" in chunks[0].text
    assert "CALLS: DGEMM" in chunks[0].text


def test_doc_comments_included():
    unit = _make_unit(doc_comments="This routine computes eigenvalues")
    chunks = chunk_units([unit])
    assert "DOCUMENTATION:" in chunks[0].text
    assert "eigenvalues" in chunks[0].text


def test_oversized_unit_splits():
    big_source = "      X = X + 1\n" * 5000
    unit = _make_unit(source_text=big_source)
    chunks = chunk_units([unit], max_tokens=500)
    assert len(chunks) > 1
    for chunk in chunks:
        assert _count_tokens(chunk.text) <= 500


def test_chunk_metadata_tracks_splits():
    big_source = "      X = X + 1\n" * 5000
    unit = _make_unit(source_text=big_source)
    chunks = chunk_units([unit], max_tokens=500)
    assert all(c.metadata["total_chunks"] == len(chunks) for c in chunks)
    indexes = [c.metadata["chunk_index"] for c in chunks]
    assert indexes == list(range(len(chunks)))


def test_empty_source():
    unit = _make_unit(source_text="")
    chunks = chunk_units([unit])
    assert len(chunks) == 1


def test_multiple_units():
    units = [_make_unit(name=f"UNIT{i}") for i in range(3)]
    chunks = chunk_units(units)
    assert len(chunks) == 3
    names = [c.metadata["unit_name"] for c in chunks]
    assert names == ["UNIT0", "UNIT1", "UNIT2"]
