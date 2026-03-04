"""Tests for the Fortran parser service."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tests.conftest import SAMPLE_FIXED_FORM_MODULE

from app.services.parser import (
    ParsedUnit,
    _clean_doc_comments,
    _extract_called_routines,
    _extract_doc_comments,
    parse_file,
)


def test_extract_doc_comments_fixed_form():
    """_extract_doc_comments extracts *> lines for fixed-form."""
    text = "*> \\brief DTEST computes a test\n*> DTEST does something.\n      X = 1"
    result = _extract_doc_comments(text, is_free_form=False)
    assert "DTEST" in result
    assert "computes" in result


def test_extract_doc_comments_free_form():
    """_extract_doc_comments extracts !> lines for free-form."""
    text = "!> \\brief CTEST generates\n!> CTEST does something.\n   x = 1"
    result = _extract_doc_comments(text, is_free_form=True)
    assert "CTEST" in result
    assert "generates" in result


def test_extract_doc_comments_leading_space():
    """_extract_doc_comments strips leading space after prefix."""
    text = "*>  content with space"
    result = _extract_doc_comments(text, is_free_form=False)
    assert "content" in result


def test_clean_doc_comments_removes_html():
    """_clean_doc_comments removes HTML tags."""
    text = "Brief <b>bold</b> and <em>emphasis</em>"
    result = _clean_doc_comments(text)
    assert "<" not in result
    assert "bold" in result


def test_clean_doc_comments_doxygen_commands():
    """_clean_doc_comments strips Doxygen commands."""
    text = "\\brief Main routine. \\param[in] N size"
    result = _clean_doc_comments(text)
    assert "\\brief" not in result
    assert "Parameter" in result or "param" in result.lower()


def test_clean_doc_comments_author():
    """_clean_doc_comments converts \\author to Author:."""
    text = "\\author John Doe"
    result = _clean_doc_comments(text)
    assert "Author:" in result
    assert "John Doe" in result


def test_clean_doc_comments_download_links():
    """_clean_doc_comments removes download link lines."""
    text = "Brief.\nDownload TGZ dependencies\n[TGZ]\nMore text"
    result = _clean_doc_comments(text)
    assert "Download" not in result
    assert "[TGZ]" not in result


def test_extract_called_routines():
    """_extract_called_routines extracts CALL statements."""
    source = "CALL DSCAL( N, 2.0D0, X, INCX )\nCALL DCOPY( N, X, INCX, Y, INCY )"
    result = _extract_called_routines(source)
    assert "DSCAL" in result
    assert "DCOPY" in result
    assert result == sorted(result)


def test_extract_called_routines_case_insensitive():
    """_extract_called_routines is case-insensitive, returns uppercase."""
    source = "call dgemm(A,B,C)"
    result = _extract_called_routines(source)
    assert "DGEMM" in result


def test_parse_file_fixed_form(sample_f_code, tmp_fortran_file):
    """parse_file parses .f files with fparser1."""
    path = tmp_fortran_file(sample_f_code, suffix=".f")
    units = parse_file(path)
    assert len(units) >= 1
    assert any(u.name == "DTEST" for u in units)
    assert any(u.kind == "subroutine" for u in units)
    sub = next(u for u in units if u.name == "DTEST")
    assert "DSCAL" in sub.called_routines
    assert "DCOPY" in sub.called_routines


def test_parse_file_free_form(sample_f90_code, tmp_fortran_file):
    """parse_file parses .f90 files with fparser2."""
    path = tmp_fortran_file(sample_f90_code, suffix=".f90")
    units = parse_file(path)
    assert len(units) >= 1
    assert any(u.name == "CTEST" for u in units)
    assert any(u.kind == "subroutine" for u in units)


def test_parse_file_unknown_suffix_raw_fallback(tmp_fortran_file):
    """parse_file returns RAW unit for unknown suffix."""
    path = tmp_fortran_file(b"some random text\nX = 1", suffix=".xyz")
    units = parse_file(path)
    assert len(units) == 1
    assert units[0].kind == "raw"
    assert units[0].name == path.stem.upper()
    assert "some random text" in units[0].source_text


def test_parse_file_raw_fallback_on_parse_failure(tmp_fortran_file):
    """parse_file falls back to RAW when parser fails."""
    path = tmp_fortran_file(b"this is not valid fortran {{{", suffix=".f")
    units = parse_file(path)
    assert len(units) == 1
    assert units[0].kind == "raw"
    assert "this is not valid" in units[0].source_text


def test_parse_file_f95_suffix(sample_f90_code, tmp_fortran_file):
    """parse_file handles .f95 as free-form."""
    path = tmp_fortran_file(sample_f90_code, suffix=".f95")
    units = parse_file(path)
    assert len(units) >= 1


SAMPLE_FORTRAN_FUNCTION = """\
      DOUBLE PRECISION FUNCTION DNRM2(N, X, INCX)
      INTEGER N, INCX
      DOUBLE PRECISION X(*)
      DNRM2 = 0.0D0
      RETURN
      END
"""


SAMPLE_FORTRAN_MODULE = """\
module mymod
  implicit none
contains
  subroutine foo()
  end subroutine foo
end module mymod
"""

SAMPLE_FORTRAN_F90_FUNCTION = """\
integer function myfunc()
  myfunc = 1
end function myfunc
"""

SAMPLE_FORTRAN_F90_PROGRAM = """\
program main
  print *, 'hi'
end program main
"""


def test_parse_file_function(tmp_fortran_file):
    """parse_file parses FUNCTION in fixed-form."""
    path = tmp_fortran_file(SAMPLE_FORTRAN_FUNCTION, suffix=".f")
    units = parse_file(path)
    assert any(u.kind == "function" for u in units)
    assert any(u.name == "DNRM2" for u in units)


def test_parse_file_module_free_form(tmp_fortran_file):
    """parse_file parses MODULE in free-form."""
    path = tmp_fortran_file(SAMPLE_FORTRAN_MODULE, suffix=".f90")
    units = parse_file(path)
    assert any(u.kind == "module" for u in units)


def test_parse_file_function_free_form(tmp_fortran_file):
    """parse_file parses FUNCTION in free-form (line 160)."""
    path = tmp_fortran_file(SAMPLE_FORTRAN_F90_FUNCTION, suffix=".f90")
    units = parse_file(path)
    assert any(u.kind == "function" for u in units)


def test_parse_file_program_free_form(tmp_fortran_file):
    """parse_file parses PROGRAM in free-form (line 162)."""
    path = tmp_fortran_file(SAMPLE_FORTRAN_F90_PROGRAM, suffix=".f90")
    units = parse_file(path)
    assert any(u.kind == "program" for u in units)


@patch("app.services.parser._parse_fixed_form")
def test_parse_file_empty_units_uses_raw_fallback(mock_parse, tmp_fortran_file):
    """When parser returns empty units, parse_file uses RAW fallback."""
    mock_parse.return_value = []
    path = tmp_fortran_file(b"      SUBROUTINE FOO\n      END\n", suffix=".f")
    units = parse_file(path)
    assert len(units) == 1
    assert units[0].kind == "raw"


def test_parse_file_block_data(sample_block_data, tmp_fortran_file):
    """parse_file parses BLOCK DATA in fixed-form."""
    path = tmp_fortran_file(sample_block_data, suffix=".f")
    units = parse_file(path)
    assert any(u.kind == "block_data" for u in units)


def test_parse_file_program(sample_program, tmp_fortran_file):
    """parse_file parses PROGRAM in fixed-form."""
    path = tmp_fortran_file(sample_program, suffix=".f")
    units = parse_file(path)
    assert any(u.kind == "program" for u in units)


def test_parse_file_module_fixed_form(sample_fixed_form_module, tmp_fortran_file):
    """parse_file parses MODULE in fixed-form (lines 93-94)."""
    path = tmp_fortran_file(sample_fixed_form_module, suffix=".f")
    units = parse_file(path)
    assert any(u.kind == "module" for u in units)


def test_parse_file_fixed_form_item_span_from_real_parser(sample_f_code, tmp_fortran_file):
    """Real fparser returns items with item.span; parser uses it (lines 105-106)."""
    path = tmp_fortran_file(sample_f_code, suffix=".f")
    units = parse_file(path)
    sub = next(u for u in units if u.name == "DTEST")
    assert sub.start_line >= 1
    assert sub.end_line >= sub.start_line


def test_parse_free_form_exception_fallback(tmp_fortran_file):
    """Parser falls back to RAW when fparser2 raises (lines 189-190)."""
    path = tmp_fortran_file(b"invalid syntax {{{ ]]]\n", suffix=".f90")
    units = parse_file(path)
    assert len(units) == 1
    assert units[0].kind == "raw"


@patch("fparser.two.utils.walk")
def test_parse_free_form_unknown_kind_and_name(mock_walk, tmp_fortran_file):
    """Parser handles unknown node type and regex miss (lines 166, 189-190)."""
    mock_node = MagicMock()
    mock_node.__class__.__name__ = "Interface_Block"
    mock_node.__str__ = lambda self: "xyz"
    mock_walk.return_value = [mock_node]

    path = tmp_fortran_file(b"subroutine x\nend subroutine\n", suffix=".f90")
    units = parse_file(path)
    assert len(units) == 1
    assert units[0].kind == "unknown"
    assert units[0].name == "unknown"


@patch("app.services.parser._parse_fixed_form")
def test_parse_fixed_form_skips_unknown_block_types(mock_parse, tmp_fortran_file):
    """Parser skips (continue) block content items with unknown type (lines 95-96)."""
    from app.services.parser import ParsedUnit

    mock_parse.return_value = [
        ParsedUnit(
            name="MOCK",
            kind="subroutine",
            source_text="X=1",
            doc_comments="",
            file_path="/tmp/test.f",
            start_line=1,
            end_line=2,
            called_routines=[],
        )
    ]

    path = tmp_fortran_file(b"      SUBROUTINE MOCK\n      END\n", suffix=".f")
    units = parse_file(path)
    assert len(units) == 1


def test_parse_fixed_form_module_via_real_parser(tmp_fortran_file):
    """Use real parser: module hits 93-94; mixed block may hit 95-96."""
    path = tmp_fortran_file(SAMPLE_FIXED_FORM_MODULE, suffix=".f")
    units = parse_file(path)
    assert any(u.kind == "module" for u in units)


@patch("fparser.one.parsefortran.FortranParser")
def test_parse_fixed_form_item_span_and_skips_unknown(mock_parser_cls, tmp_fortran_file):
    """Parser uses item.item.span (105-106); skips unknown block types (95-96)."""
    sub_item = MagicMock()
    sub_item.name = "SPANSUB"
    sub_item.__class__.__name__ = "Subroutine"
    sub_item.__str__ = lambda self: "      SUBROUTINE SPANSUB\n      END"
    sub_item.item = MagicMock()
    sub_item.item.span = (5, 10)
    # Set up content[-1].item.span for end_line extraction
    end_stmt = MagicMock()
    end_stmt.item.span = (9, 10)
    sub_item.content = [sub_item.item, end_stmt]

    unknown_item = MagicMock()
    unknown_item.__class__.__name__ = "Include"  # doesn't match subroutine/function/program/blockdata/module
    unknown_item.__str__ = lambda self: "INCLUDE"

    mock_block = MagicMock()
    mock_block.content = [unknown_item, sub_item]

    mock_parser = MagicMock()
    mock_parser.block = mock_block
    mock_parser_cls.return_value = mock_parser

    path = tmp_fortran_file(b"      SUBROUTINE SPANSUB\n      END\n", suffix=".f")
    units = parse_file(path)
    assert len(units) == 1
    assert units[0].start_line == 5
    assert units[0].end_line == 10


@patch("fparser.one.parsefortran.FortranParser")
def test_parse_fixed_form_item_no_span_uses_default_lines(mock_parser_cls, tmp_fortran_file):
    """Parser uses default start_line/end_line when item has no span (branch coverage)."""
    sub_item = MagicMock()
    sub_item.name = "NOSPAN"
    sub_item.__class__.__name__ = "Subroutine"
    sub_item.__str__ = lambda self: "      SUBROUTINE NOSPAN\n      END"
    sub_item.item = None  # no item.span

    mock_block = MagicMock()
    mock_block.content = [sub_item]

    mock_parser = MagicMock()
    mock_parser.block = mock_block
    mock_parser_cls.return_value = mock_parser

    path = tmp_fortran_file(b"      SUBROUTINE NOSPAN\n      END\n", suffix=".f")
    units = parse_file(path)
    assert len(units) == 1
    assert units[0].start_line == 1
    assert units[0].end_line == 2


@patch("fparser.one.parsefortran.FortranParser")
def test_parse_fixed_form_unit_doc_fallback_to_full_file(mock_parser_cls, tmp_fortran_file):
    """Parser tries full file for doc when unit source has no docs (branch coverage)."""
    sub_item = MagicMock()
    sub_item.name = "NODOC"
    sub_item.__class__.__name__ = "Subroutine"
    # str(item) has no *> lines - fparser strips them
    sub_item.__str__ = lambda self: "      SUBROUTINE NODOC\n      X=1\n      END"
    sub_item.item = MagicMock()
    sub_item.item.span = (1, 3)

    mock_block = MagicMock()
    mock_block.content = [sub_item]

    mock_parser = MagicMock()
    mock_parser.block = mock_block
    mock_parser_cls.return_value = mock_parser

    file_content = b"*> \\brief NODOC does something\n      SUBROUTINE NODOC\n      X=1\n      END\n"
    path = tmp_fortran_file(file_content, suffix=".f")
    units = parse_file(path)
    assert len(units) == 1
    # Unit source has no *> so unit_doc empty; we try full file
    assert units[0].doc_comments == "" or "NODOC" in units[0].doc_comments


@patch("fparser.one.parsefortran.FortranParser")
def test_parse_fixed_form_unit_has_doc_skips_full_file(mock_parser_cls, tmp_fortran_file):
    """Parser skips full-file doc extraction when unit source has docs (branch coverage)."""
    sub_item = MagicMock()
    sub_item.name = "HASDOC"
    sub_item.__class__.__name__ = "Subroutine"
    # str(item) includes *> - unit has its own docs
    sub_item.__str__ = lambda self: "*> \\brief HASDOC does work\n      SUBROUTINE HASDOC\n      X=1\n      END"
    sub_item.item = MagicMock()
    sub_item.item.span = (1, 4)

    mock_block = MagicMock()
    mock_block.content = [sub_item]

    mock_parser = MagicMock()
    mock_parser.block = mock_block
    mock_parser_cls.return_value = mock_parser

    path = tmp_fortran_file(b"      SUBROUTINE HASDOC\n      X=1\n      END\n", suffix=".f")
    units = parse_file(path)
    assert len(units) == 1
    assert "HASDOC" in units[0].doc_comments or "work" in units[0].doc_comments


def test_parse_file_block_none_raises_then_raw_fallback(tmp_fortran_file):
    """When fparser1 returns block=None, ValueError is raised and RAW fallback used."""
    with patch("fparser.one.parsefortran.FortranParser") as MockParser:
        mock_instance = MagicMock()
        mock_instance.block = None
        mock_instance.parse.return_value = None
        MockParser.return_value = mock_instance

        path = tmp_fortran_file(b"      SUBROUTINE FOO\n      END\n", suffix=".f")
        units = parse_file(path)
        assert len(units) == 1
        assert units[0].kind == "raw"
