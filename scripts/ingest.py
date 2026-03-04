"""CLI ingestion script: parse → chunk → embed → upsert."""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.services.parser import parse_file
from app.services.chunker import chunk_units
from app.services.embeddings import embed_texts
from app.services.vector_store import ensure_collection, upsert_chunks, delete_collection
from app.embedding_registry import EMBEDDING_MODELS, collection_name_for_model, get_model_info

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
    parser.add_argument("--embedding-model", type=str, default=None,
                        help="Embedding model to use (default: from settings)")
    parser.add_argument("--collection-name", type=str, default=None,
                        help="Qdrant collection name override")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete target collection before ingesting")
    parser.add_argument("--list-models", action="store_true",
                        help="List all registered embedding models and exit")
    args = parser.parse_args()

    if args.list_models:
        print(f"{'Model':<30} {'Provider':<10} {'Dims':>6} {'Max Tokens':>11}")
        print("-" * 60)
        for name, info in EMBEDDING_MODELS.items():
            print(f"{name:<30} {info.provider:<10} {info.dimensions:>6} {info.max_tokens:>11}")
        return

    # Resolve embedding model and collection name
    embedding_model = args.embedding_model
    if embedding_model and embedding_model in EMBEDDING_MODELS:
        model_info = get_model_info(embedding_model)
        embedding_dim = model_info.dimensions
        target_collection = args.collection_name or collection_name_for_model("lapack", embedding_model)
        logger.info("Using embedding model: %s (dim=%d, collection=%s)",
                    embedding_model, embedding_dim, target_collection)
    else:
        embedding_dim = None  # use settings default
        target_collection = args.collection_name  # None = use settings default
        if embedding_model:
            logger.warning("Unknown embedding model '%s', using settings default", embedding_model)

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

    # Build reverse call-graph (called_by_map)
    called_by_map: dict[str, list[str]] = {}
    for unit in all_units:
        for called in unit.called_routines:
            called_by_map.setdefault(called, []).append(unit.name)
    logger.info("Built reverse call-graph: %d routines have callers", len(called_by_map))

    # Chunk all units
    all_chunks = chunk_units(all_units, called_by_map=called_by_map)
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

    # Handle --recreate
    if args.recreate:
        coll_to_delete = target_collection or get_settings().QDRANT_COLLECTION_NAME
        if delete_collection(coll_to_delete):
            logger.info("Deleted collection '%s' for recreate", coll_to_delete)
        # Also delete legacy "lapack" collection if it exists and is different
        if coll_to_delete != "lapack":
            if delete_collection("lapack"):
                logger.info("Deleted legacy 'lapack' collection")

    # Ensure collection exists
    ensure_collection(collection_name=target_collection, embedding_dim=embedding_dim)

    # Embed and upsert in batches
    batch_size = args.batch_size
    total_upserted = 0

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c.text for c in batch]

        logger.info("Embedding batch %d-%d of %d...",
                    i, i + len(batch), len(all_chunks))
        embeddings = embed_texts(texts, model=embedding_model)

        if len(embeddings) != len(batch):
            logger.error("Embedding count mismatch: %d embeddings for %d chunks — skipping batch",
                        len(embeddings), len(batch))
            continue

        logger.info("Upserting batch...")
        upsert_chunks(batch, embeddings, collection_name=target_collection)
        total_upserted += len(batch)

        logger.info("Progress: %d/%d chunks upserted", total_upserted, len(all_chunks))

    elapsed = time.time() - start_time
    chunks_per_sec = total_upserted / elapsed if elapsed > 0 else 0
    logger.info("Ingestion complete: %d chunks upserted in %.1fs (%.1f chunks/sec)", total_upserted, elapsed, chunks_per_sec)


if __name__ == "__main__":
    main()
