"""Fortran source file parser using fparser1 (.f) and fparser2 (.f90)."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ParsedUnit:
    name: str
    kind: str  # subroutine, function, program, module, block_data, raw
    source_text: str
    doc_comments: str
    file_path: str
    start_line: int
    end_line: int
    called_routines: list[str] = field(default_factory=list)


def _extract_doc_comments(text: str, is_free_form: bool) -> str:
    """Extract Doxygen-style doc comments (*> for fixed-form, !> for free-form)."""
    prefix = "!>" if is_free_form else "*>"
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            content = stripped[len(prefix):]
            if content.startswith(" "):
                content = content[1:]
            lines.append(content)
    raw = "\n".join(lines)
    return _clean_doc_comments(raw)


def _clean_doc_comments(text: str) -> str:
    """Strip Doxygen commands, HTML tags, and download links from doc comments."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove Doxygen commands that add no content
    # Remove Doxygen commands — order matters: \param before \par
    text = re.sub(r'\\(?:addtogroup|ingroup|endverbatim|verbatim)\s*\w*', '', text)
    text = re.sub(r'\\param\[(?:in|out|in,out)\]\s*', '\nParameter ', text)
    text = re.sub(r'\\(?:par|brief|b)\b\s*', '', text)
    text = re.sub(r'\\author\s*', 'Author: ', text)
    # Remove download link lines
    text = re.sub(r'Download .+ dependencies\n?', '', text)
    text = re.sub(r'\[(?:TGZ|ZIP|TXT)\]\n?', '', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _extract_called_routines(source: str) -> list[str]:
    """Extract CALL statements from Fortran source."""
    calls = set()
    for match in re.finditer(r'\bCALL\s+(\w+)', source, re.IGNORECASE):
        calls.add(match.group(1).upper())
    return sorted(calls)


def _parse_fixed_form(path: Path) -> list[ParsedUnit]:
    """Parse fixed-form Fortran (.f) using fparser1."""
    from fparser.one.parsefortran import FortranParser
    from fparser.common.readfortran import FortranFileReader

    units = []
    text = path.read_text(errors="replace")
    file_str = str(path)

    try:
        reader = FortranFileReader(str(path))
        parser = FortranParser(reader)
        parser.parse()
        block = parser.block

        if block is None:
            raise ValueError("Parser returned None block")

        for item in block.content:
            type_name = type(item).__name__.lower()

            if "subroutine" in type_name:
                kind = "subroutine"
            elif "function" in type_name:
                kind = "function"
            elif "program" in type_name:
                kind = "program"
            elif "blockdata" in type_name or "block_data" in type_name:
                kind = "block_data"
            elif "module" in type_name:
                kind = "module"
            else:
                continue

            name_attr = getattr(item, "name", None) or "unknown"
            source_text = str(item)

            start_line = 1
            end_line = len(text.splitlines())

            # Try to find actual line range
            if hasattr(item, "item") and hasattr(item.item, "span"):
                start_line = item.item.span[0]
                try:
                    end_line = item.content[-1].item.span[1]
                except (AttributeError, IndexError, TypeError):
                    end_line = item.item.span[1]

            # Extract doc comments from the unit's source text only
            unit_doc = _extract_doc_comments(source_text, is_free_form=False)
            # If the unit source doesn't have docs, try the full file
            # (fparser1 may strip comments from str(item))
            if not unit_doc:
                unit_doc = _extract_doc_comments(text, is_free_form=False)

            units.append(ParsedUnit(
                name=str(name_attr).upper(),
                kind=kind,
                source_text=source_text,
                doc_comments=unit_doc,
                file_path=file_str,
                start_line=start_line,
                end_line=end_line,
                called_routines=_extract_called_routines(source_text),
            ))
    except Exception as e:
        logger.warning("fparser1 parse failed for %s: %s — will use RAW fallback", path, e)

    return units


def _get_fparser2_span(node) -> tuple[int, int] | None:
    """Extract (start_line, end_line) from an fparser2 AST node's content items."""
    try:
        content = getattr(node, "content", None)
        if not content:
            return None
        from fparser.two import Fortran2003
        # Filter out Comment nodes
        items = [item for item in content if not isinstance(item, Fortran2003.Comment)]
        if not items:
            return None
        start_line = items[0].item.span[0]
        end_line = items[-1].item.span[1]
        return (start_line, end_line)
    except (AttributeError, IndexError, TypeError):
        return None


def _parse_free_form(path: Path) -> list[ParsedUnit]:
    """Parse free-form Fortran (.f90) using fparser2."""
    from fparser.two.parser import ParserFactory
    from fparser.two.utils import walk
    from fparser.two import Fortran2003
    from fparser.common.readfortran import FortranFileReader

    units = []
    text = path.read_text(errors="replace")
    file_str = str(path)

    try:
        reader = FortranFileReader(str(path))
        f2003_parser = ParserFactory().create(std="f2003")
        tree = f2003_parser(reader)

        target_types = (
            Fortran2003.Subroutine_Subprogram,
            Fortran2003.Function_Subprogram,
            Fortran2003.Main_Program,
            Fortran2003.Module,
        )

        for node in walk(tree, target_types):
            type_name = type(node).__name__

            if "Subroutine" in type_name:
                kind = "subroutine"
            elif "Function" in type_name:
                kind = "function"
            elif "Program" in type_name:
                kind = "program"
            elif "Module" in type_name:
                kind = "module"
            else:
                kind = "unknown"

            source_text = str(node)

            # Extract name via regex (most reliable across fparser2 versions)
            name = "unknown"
            m = re.search(
                r'(?:SUBROUTINE|FUNCTION|PROGRAM|MODULE)\s+(\w+)',
                source_text, re.IGNORECASE
            )
            if m:
                name = m.group(1).upper()

            span = _get_fparser2_span(node)
            if span:
                start_line, end_line = span
            else:
                start_line, end_line = 1, len(text.splitlines())

            units.append(ParsedUnit(
                name=name,
                kind=kind,
                source_text=source_text,
                doc_comments=_extract_doc_comments(source_text, is_free_form=True),
                file_path=file_str,
                start_line=start_line,
                end_line=end_line,
                called_routines=_extract_called_routines(source_text),
            ))
    except Exception as e:
        logger.warning("fparser2 failed for %s: %s — will use RAW fallback", path, e)

    return units


def parse_file(path: Path) -> list[ParsedUnit]:
    """Parse a Fortran file and return parsed units. Falls back to RAW on failure."""
    text = path.read_text(errors="replace")
    file_str = str(path)
    suffix = path.suffix.lower()

    if suffix in (".f90", ".f95", ".f03", ".f08"):
        units = _parse_free_form(path)
    elif suffix in (".f", ".for", ".fpp"):
        units = _parse_fixed_form(path)
    else:
        units = []

    # RAW fallback: if no units extracted, return the whole file as one unit
    if not units:
        name = path.stem.upper()
        units = [ParsedUnit(
            name=name,
            kind="raw",
            source_text=text,
            doc_comments=_extract_doc_comments(
                text, is_free_form=suffix in (".f90", ".f95", ".f03", ".f08")
            ),
            file_path=file_str,
            start_line=1,
            end_line=len(text.splitlines()),
            called_routines=_extract_called_routines(text),
        )]

    return units
