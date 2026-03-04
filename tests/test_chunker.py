"""Tests for the chunker service."""

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


def test_extract_purpose_via_chunk_units():
    """Doc comments with routine name produce PURPOSE line in header."""
    doc = "DTEST computes the LU factorization of a matrix."
    unit = _make_unit(doc_comments=doc, name="DTEST")
    chunks = chunk_units([unit])
    assert "PURPOSE:" in chunks[0].text
    assert "LU factorization" in chunks[0].text


def test_extract_purpose_stops_at_parameter():
    """Doc with Parameter line is included; purpose extraction stops before it."""
    doc = "Purpose: Main routine.\nParameter N - input size"
    unit = _make_unit(doc_comments=doc)
    chunks = chunk_units([unit])
    assert "Main routine" in chunks[0].text


def test_sliding_window_small_max_tokens():
    """Oversized unit with small max_tokens uses sliding window split."""
    big_source = "      X = X + 1\n" * 200
    unit = _make_unit(source_text=big_source)
    chunks = chunk_units([unit], max_tokens=150)
    assert len(chunks) > 1
    for c in chunks:
        assert _count_tokens(c.text) <= 150


def test_header_truncation_when_doc_too_large():
    """When doc_comments make header huge, truncate to header-only then split."""
    huge_doc = "Doc line.\n" * 500
    big_source = "      X = X + 1\n" * 500
    unit = _make_unit(source_text=big_source, doc_comments=huge_doc)
    chunks = chunk_units([unit], max_tokens=200)
    assert len(chunks) >= 1
    for c in chunks:
        assert _count_tokens(c.text) <= 200


def test_extract_purpose_blank_line_breaks():
    """Empty line after purpose content causes break."""
    doc = "Purpose: Main text.\n\nMore after blank"
    unit = _make_unit(doc_comments=doc)
    chunks = chunk_units([unit])
    assert "Main text" in chunks[0].text


def test_extract_purpose_skips_routine_name_only_line():
    """Line that is just the routine name is skipped."""
    doc = "DTEST\nDTEST does something useful."
    unit = _make_unit(doc_comments=doc, name="DTEST")
    chunks = chunk_units([unit])
    assert "something useful" in chunks[0].text


def test_extract_purpose_author_stops():
    """Author: line stops purpose extraction."""
    doc = "Purpose: Main.\nAuthor: Jane"
    unit = _make_unit(doc_comments=doc)
    chunks = chunk_units([unit])
    assert "Main" in chunks[0].text


def test_extract_purpose_blank_lines_continue():
    """Blank lines at start trigger continue (line 40)."""
    doc = "\n\nDTEST computes eigenvalues."
    unit = _make_unit(doc_comments=doc, name="DTEST")
    chunks = chunk_units([unit])
    assert "eigenvalues" in chunks[0].text


def test_extract_purpose_parameter_line_continue():
    """Parameter line when no purpose yet triggers continue (line 45)."""
    doc = "Parameter N - size\nDTEST does the work."
    unit = _make_unit(doc_comments=doc, name="DTEST")
    chunks = chunk_units([unit])
    assert "work" in chunks[0].text


def test_extract_purpose_blank_after_content_breaks():
    """Blank line after purpose content triggers break (line 40)."""
    doc = "Purpose: Main.\nSome text here\n\nMore after blank"
    unit = _make_unit(doc_comments=doc)
    chunks = chunk_units([unit])
    assert "PURPOSE:" in chunks[0].text
    assert "Some text here" in chunks[0].text


def test_extract_purpose_parameter_after_content_breaks():
    """Parameter line after purpose content triggers break (line 45)."""
    doc = "Purpose: Main.\nSome text\nParameter N - input"
    unit = _make_unit(doc_comments=doc)
    chunks = chunk_units([unit])
    assert "PURPOSE:" in chunks[0].text
    assert "Some text" in chunks[0].text


def test_sliding_window_empty_source():
    """_sliding_window_split with empty text returns single empty chunk."""
    from app.services.chunker import _sliding_window_split

    result = _sliding_window_split("", max_tokens=50)
    assert result == [("", 0)]


def test_sliding_window_single_chunk_breaks_loop():
    """_sliding_window_split breaks when end >= len(tokens) (branch coverage)."""
    from app.services.chunker import _sliding_window_split

    short = "Hello world"
    result = _sliding_window_split(short, max_tokens=100)
    assert len(result) == 1
    assert result[0] == (short, 0)


def test_sliding_window_min_tokens_clamp():
    """_sliding_window_split clamps max_tokens to MIN_WINDOW_TOKENS when smaller."""
    from app.services.chunker import _sliding_window_split

    big = "word " * 500
    result = _sliding_window_split(big, max_tokens=50)
    assert len(result) >= 1
    for text, tok_offset in result:
        assert len(text) > 0


def test_sliding_window_multiple_iterations():
    """_sliding_window_split continues loop when end < len(tokens) (branch coverage)."""
    from app.services.chunker import _sliding_window_split, _count_tokens

    big = "word " * 800
    result = _sliding_window_split(big, max_tokens=80)
    assert len(result) >= 3
    for text, tok_offset in result:
        assert _count_tokens(text) <= 100


def test_safety_truncation_when_chunk_exceeds_limit():
    """Chunk text exceeding max_tokens gets truncated."""
    from app.services.chunker import _count_tokens

    long_line = "      " + "CALL FOO(BAR,BAZ)\n" * 80
    unit = _make_unit(source_text=long_line)
    chunks = chunk_units([unit], max_tokens=120)
    for c in chunks:
        assert _count_tokens(c.text) <= 120
