"""CLI ingestion script: parse → chunk → embed → upsert."""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.parser import parse_file
from app.services.chunker import chunk_units
from app.services.embeddings import embed_texts
from app.services.vector_store import ensure_collection, upsert_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


EXCLUDE_DIRS = {"VARIANTS", "TESTING", "INSTALL", "CBLAS"}


def find_fortran_files(data_dir: Path, extensions: list[str]) -> list[Path]:
    """Recursively find Fortran source files, excluding test/variant dirs."""
    files = []
    for ext in extensions:
        for f in data_dir.rglob(f"*{ext}"):
            if not any(part in EXCLUDE_DIRS for part in f.parts):
                files.append(f)
    # Sort for deterministic ordering
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Ingest Fortran source files into LegacyLens")
    parser.add_argument("--data-dir", type=str, default="data/lapack",
                        help="Root directory containing Fortran source files")
    parser.add_argument("--extensions", type=str, nargs="+",
                        default=[".f", ".f90"],
                        help="File extensions to process")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Number of chunks to embed per batch")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and chunk only, don't embed or upsert")
    parser.add_argument("--subdirs", type=str, nargs="+",
                        default=["SRC", "BLAS/SRC"],
                        help="Subdirectories to ingest")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    start_time = time.time()

    # Collect files from specified subdirectories
    all_files = []
    for subdir in args.subdirs:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            files = find_fortran_files(subdir_path, args.extensions)
            all_files.extend(files)
            logger.info("Found %d files in %s", len(files), subdir_path)
        else:
            logger.warning("Subdirectory not found: %s", subdir_path)

    logger.info("Total files to process: %d", len(all_files))

    # Parse all files
    all_units = []
    parse_errors = 0
    for i, file_path in enumerate(all_files):
        try:
            units = parse_file(file_path)
            all_units.extend(units)
            if (i + 1) % 100 == 0:
                logger.info("Parsed %d/%d files (%d units so far)",
                            i + 1, len(all_files), len(all_units))
        except Exception as e:
            parse_errors += 1
            logger.error("Failed to parse %s: %s", file_path, e)

    logger.info("Parsing complete: %d units from %d files (%d errors)",
                len(all_units), len(all_files), parse_errors)

    # Chunk all units
    all_chunks = chunk_units(all_units)
    logger.info("Chunking complete: %d chunks", len(all_chunks))

    if args.dry_run:
        logger.info("DRY RUN — skipping embedding and upsert")
        logger.info("Summary: %d files → %d units → %d chunks",
                    len(all_files), len(all_units), len(all_chunks))
        # Print sample
        if all_chunks:
            sample = all_chunks[0]
            logger.info("Sample chunk metadata: %s", sample.metadata)
            logger.info("Sample chunk text (first 200 chars): %s", sample.text[:200])
        elapsed = time.time() - start_time
        logger.info("Dry run completed in %.1fs", elapsed)
        return

    # Ensure collection exists
    ensure_collection()

    # Embed and upsert in batches
    batch_size = args.batch_size
    total_upserted = 0

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c.text for c in batch]

        logger.info("Embedding batch %d-%d of %d...",
                    i, i + len(batch), len(all_chunks))
        embeddings = embed_texts(texts)

        if len(embeddings) != len(batch):
            logger.error("Embedding count mismatch: %d embeddings for %d chunks — skipping batch",
                        len(embeddings), len(batch))
            continue

        logger.info("Upserting batch...")
        upsert_chunks(batch, embeddings)
        total_upserted += len(batch)

        logger.info("Progress: %d/%d chunks upserted", total_upserted, len(all_chunks))

    elapsed = time.time() - start_time
    logger.info("Ingestion complete: %d chunks upserted in %.1fs", total_upserted, elapsed)


if __name__ == "__main__":
    main()
