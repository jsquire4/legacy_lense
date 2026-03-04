"""Custom chunking with token cap enforcement via tiktoken."""

import logging
from dataclasses import dataclass

import tiktoken

from app.services.parser import ParsedUnit

logger = logging.getLogger(__name__)

_encoder = tiktoken.get_encoding("cl100k_base")

DEFAULT_MAX_TOKENS = 8191
WINDOW_OVERLAP = 50
MIN_WINDOW_TOKENS = 100


@dataclass
class Chunk:
    text: str
    metadata: dict


def _count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def _extract_purpose(doc_comments: str, name: str) -> str:
    """Extract the first meaningful purpose sentence from doc comments."""
    if not doc_comments:
        return ""
    lines = doc_comments.splitlines()
    purpose_lines = []
    in_purpose = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_purpose and purpose_lines:
                break
            continue
        # Stop at parameter descriptions, author lines, or separator lines
        if stripped.startswith(("Parameter ", "Author:", "==")):
            if in_purpose and purpose_lines:
                break
            continue
        # Skip lines that are just the routine name
        if stripped.upper() == name:
            continue
        # "Purpose:" header signals the start
        if "Purpose:" in stripped:
            in_purpose = True
            continue
        # First line with the routine name is the brief
        if name in stripped.upper() and not in_purpose:
            purpose_lines.append(stripped)
            in_purpose = True
            continue
        if in_purpose:
            purpose_lines.append(stripped)
    result = " ".join(purpose_lines).strip()
    # Clean up leftover Doxygen artifacts
    result = result.replace("  ", " ")
    return result[:500]


def _build_metadata_header(unit: ParsedUnit, called_by: list[str] | None = None,
                           start_line: int | None = None, end_line: int | None = None) -> str:
    """Build a structured metadata header for embedding context."""
    from pathlib import Path
    filename = Path(unit.file_path).name

    # Front-load routine identity and purpose
    purpose = _extract_purpose(unit.doc_comments, unit.name)

    lines = [
        f"ROUTINE: {unit.name} ({unit.kind})",
        f"FILE: {filename}",
    ]
    if purpose:
        lines.append(f"PURPOSE: {purpose}")
    if unit.called_routines:
        lines.append(f"CALLS: {', '.join(unit.called_routines)}")
    if called_by:
        lines.append(f"CALLED_BY: {', '.join(called_by[:10])}")
    lines.append(f"LINES: {start_line or unit.start_line}-{end_line or unit.end_line}")
    return "\n".join(lines)


def _build_metadata(unit: ParsedUnit, chunk_index: int = 0, total_chunks: int = 1,
                    chunk_start_line: int | None = None, chunk_end_line: int | None = None,
                    called_by: list[str] | None = None) -> dict:
    meta = {
        "file_path": unit.file_path,
        "unit_type": unit.kind,
        "unit_name": unit.name,
        "start_line": chunk_start_line if chunk_start_line is not None else unit.start_line,
        "end_line": chunk_end_line if chunk_end_line is not None else unit.end_line,
        "called_routines": unit.called_routines,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }
    if called_by:
        meta["called_by"] = called_by
    return meta


def _sliding_window_split(text: str, max_tokens: int) -> list[tuple[str, int]]:
    """Split text into overlapping windows of max_tokens tokens.

    Returns list of (window_text, token_start_index) tuples so callers can
    compute exact line offsets without unreliable str.find().
    """
    if max_tokens < MIN_WINDOW_TOKENS:
        max_tokens = MIN_WINDOW_TOKENS

    tokens = _encoder.encode(text)
    if not tokens:
        return [(text, 0)]

    chunks = []
    start = 0
    step = max(1, max_tokens - WINDOW_OVERLAP)
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append((_encoder.decode(chunk_tokens), start))
        if end >= len(tokens):
            break
        start += step
    return chunks


def chunk_units(units: list[ParsedUnit], max_tokens: int = DEFAULT_MAX_TOKENS,
                called_by_map: dict[str, list[str]] | None = None) -> list[Chunk]:
    """Convert parsed units into chunks, enforcing token cap."""
    chunks = []

    for unit in units:
        callers = called_by_map.get(unit.name, []) if called_by_map else []
        header = _build_metadata_header(unit, called_by=callers or None)
        doc_section = ""
        if unit.doc_comments:
            doc_section = f"\nDOCUMENTATION:\n{unit.doc_comments}\n"

        full_text = f"{header}{doc_section}\n\nSOURCE:\n{unit.source_text}"
        header_with_doc = f"{header}{doc_section}\n\nSOURCE:\n"
        header_tokens = _count_tokens(header_with_doc)

        if _count_tokens(full_text) <= max_tokens:
            chunks.append(Chunk(
                text=full_text,
                metadata=_build_metadata(unit, called_by=callers or None),
            ))
        else:
            # Need to split the source text with sliding window
            available_tokens = max_tokens - header_tokens
            if available_tokens < MIN_WINDOW_TOKENS:
                # Header alone is too large — truncate doc comments
                header_with_doc = f"{header}\n\nSOURCE:\n"
                header_tokens = _count_tokens(header_with_doc)
                available_tokens = max_tokens - header_tokens

            source_windows = _sliding_window_split(unit.source_text, available_tokens)
            total = len(source_windows)

            # Pre-compute cumulative newline counts per token so we can map
            # any token offset to a line number in O(1).
            source_tokens = _encoder.encode(unit.source_text)
            # newlines_before[i] = number of newlines in tokens[0..i-1]
            cumulative_newlines: list[int] = []
            running = 0
            for tok in source_tokens:
                cumulative_newlines.append(running)
                running += _encoder.decode([tok]).count("\n")

            for i, (window, tok_start) in enumerate(source_windows):
                # Exact line offset from token position — no str.find() needed
                line_offset = cumulative_newlines[tok_start] if tok_start < len(cumulative_newlines) else 0
                chunk_start_line = unit.start_line + line_offset
                chunk_end_line = chunk_start_line + window.count("\n")

                # Rebuild header with per-chunk line range
                chunk_header = _build_metadata_header(unit, called_by=callers or None,
                                                      start_line=chunk_start_line,
                                                      end_line=chunk_end_line)
                chunk_header_with_doc = f"{chunk_header}{doc_section}\n\nSOURCE:\n"
                chunk_text = f"{chunk_header_with_doc}{window}"

                # Final safety truncation
                if _count_tokens(chunk_text) > max_tokens:
                    toks = _encoder.encode(chunk_text)[:max_tokens]
                    chunk_text = _encoder.decode(toks)

                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=_build_metadata(unit, chunk_index=i, total_chunks=total,
                                             chunk_start_line=chunk_start_line,
                                             chunk_end_line=chunk_end_line,
                                             called_by=callers or None),
                ))

            logger.info(
                "Split %s:%s into %d chunks (source was %d tokens)",
                unit.file_path, unit.name, total, _count_tokens(unit.source_text),
            )

    return chunks
