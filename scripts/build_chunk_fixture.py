"""One-time script: parse all Fortran files and serialize chunks to a gzipped JSONL fixture.

Run locally whenever the LAPACK source changes (rarely):
    python scripts/build_chunk_fixture.py
"""

import gzip
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.parser import parse_file
from app.services.chunker import chunk_units

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

EXCLUDE_DIRS = {"VARIANTS", "TESTING", "INSTALL", "CBLAS"}
DEFAULT_DATA_DIR = Path("data/lapack")
DEFAULT_SUBDIRS = ["SRC", "BLAS/SRC"]
DEFAULT_EXTENSIONS = [".f", ".f90"]
OUTPUT_PATH = Path("fixtures/chunks.jsonl.gz")


def find_fortran_files(data_dir: Path, extensions: list[str], subdirs: list[str]) -> list[Path]:
    """Recursively find Fortran source files, excluding test/variant dirs."""
    files = []
    for subdir in subdirs:
        subdir_path = data_dir / subdir
        if not subdir_path.exists():
            logger.warning("Subdirectory not found: %s", subdir_path)
            continue
        for ext in extensions:
            for f in subdir_path.rglob(f"*{ext}"):
                if not any(part in EXCLUDE_DIRS for part in f.parts):
                    files.append(f)
    return sorted(files)


def main():
    data_dir = DEFAULT_DATA_DIR
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    t0 = time.time()

    # Find files
    all_files = find_fortran_files(data_dir, DEFAULT_EXTENSIONS, DEFAULT_SUBDIRS)
    logger.info("Found %d Fortran files", len(all_files))

    # Parse all files
    all_units = []
    parse_errors = 0
    for i, file_path in enumerate(all_files):
        try:
            units = parse_file(file_path)
            all_units.extend(units)
            if (i + 1) % 100 == 0:
                logger.info("Parsed %d/%d files (%d units so far)", i + 1, len(all_files), len(all_units))
        except Exception as e:
            parse_errors += 1
            logger.error("Failed to parse %s: %s", file_path, e)

    logger.info("Parsing complete: %d units from %d files (%d errors)", len(all_units), len(all_files), parse_errors)

    # Build reverse call-graph
    called_by_map: dict[str, list[str]] = {}
    for unit in all_units:
        for called in unit.called_routines:
            called_by_map.setdefault(called, []).append(unit.name)
    logger.info("Built reverse call-graph: %d routines have callers", len(called_by_map))

    # Chunk all units
    all_chunks = chunk_units(all_units, called_by_map=called_by_map)
    logger.info("Chunking complete: %d chunks", len(all_chunks))

    # Write gzipped JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as f:
        for chunk in all_chunks:
            line = json.dumps({"text": chunk.text, "metadata": chunk.metadata}, ensure_ascii=False)
            f.write(line + "\n")

    size_bytes = OUTPUT_PATH.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    elapsed = time.time() - t0

    logger.info("Fixture written to %s", OUTPUT_PATH)
    logger.info("Stats: %d files → %d units → %d chunks", len(all_files), len(all_units), len(all_chunks))
    logger.info("Output size: %.2f MB (%.0f bytes)", size_mb, size_bytes)
    logger.info("Completed in %.1fs", elapsed)


if __name__ == "__main__":
    main()
