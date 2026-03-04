"""Ingestion benchmarking SSE stream generator."""

import asyncio
import logging
import time
from pathlib import Path

from app.embedding_registry import EMBEDDING_MODELS, collection_name_for_model, get_model_info
from app.services.parser import parse_file
from app.services.chunker import chunk_units
from app.services.embeddings import embed_texts
from app.services.vector_store import ensure_collection, upsert_chunks, delete_collection
from app.config import get_settings
from app.sse import sse_event as _sse_event

logger = logging.getLogger(__name__)

EXCLUDE_DIRS = {"VARIANTS", "TESTING", "INSTALL", "CBLAS"}
_DEFAULT_EXTENSIONS = [".f", ".f90"]
_DEFAULT_SUBDIRS = ["SRC", "BLAS/SRC"]
_EMBED_BATCH_SIZE = 50

# Concurrency guard — only one ingestion at a time
_ingest_lock = asyncio.Lock()


def _find_fortran_files(data_dir: Path, extensions: list[str], subdirs: list[str]) -> list[Path]:
    """Recursively find Fortran source files, excluding test/variant dirs."""
    files = []
    for subdir in subdirs:
        subdir_path = data_dir / subdir
        if not subdir_path.exists():
            continue
        for ext in extensions:
            for f in subdir_path.rglob(f"*{ext}"):
                if not any(part in EXCLUDE_DIRS for part in f.parts):
                    files.append(f)
    return sorted(files)


async def ingest_stream_generator(embedding_model: str):
    """Async generator that streams ingestion progress as SSE events."""
    if embedding_model not in EMBEDDING_MODELS:
        yield _sse_event("error", {"message": f"Unknown embedding model: {embedding_model}"})
        return

    if _ingest_lock.locked():
        yield _sse_event("error", {"message": "An ingestion is already running. Please wait."})
        return

    async with _ingest_lock:
        settings = get_settings()
        data_dir = Path(settings.DATA_DIR)

        if not data_dir.exists():
            yield _sse_event("error", {"message": f"Data directory not found: {data_dir}"})
            return

        model_info = get_model_info(embedding_model)
        target_collection = collection_name_for_model("lapack", embedding_model)

        t0 = time.time()

        # Phase 1: Parse files
        all_files = await asyncio.to_thread(
            _find_fortran_files, data_dir, _DEFAULT_EXTENSIONS, _DEFAULT_SUBDIRS
        )
        if not all_files:
            yield _sse_event("error", {"message": f"No Fortran files found in {data_dir}"})
            return

        all_units = []
        for file_path in all_files:
            try:
                units = await asyncio.to_thread(parse_file, file_path)
                all_units.extend(units)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", file_path, e)

        # Build reverse call-graph
        called_by_map: dict[str, list[str]] = {}
        for unit in all_units:
            for called in unit.called_routines:
                called_by_map.setdefault(called, []).append(unit.name)

        all_chunks = await asyncio.to_thread(chunk_units, all_units, called_by_map=called_by_map)

        yield _sse_event("progress", {
            "phase": "parsing",
            "files": len(all_files),
            "units": len(all_units),
            "chunks": len(all_chunks),
        })

        # Phase 2: Delete old collection and recreate
        await asyncio.to_thread(delete_collection, target_collection)
        await asyncio.to_thread(
            ensure_collection,
            collection_name=target_collection,
            embedding_dim=model_info.dimensions,
        )

        # Phase 3: Embed and upsert in batches
        total_chunks = len(all_chunks)
        total_upserted = 0

        for i in range(0, total_chunks, _EMBED_BATCH_SIZE):
            batch = all_chunks[i:i + _EMBED_BATCH_SIZE]
            texts = [c.text for c in batch]

            batch_t0 = time.time()
            embeddings = await asyncio.to_thread(embed_texts, texts, embedding_model)
            await asyncio.to_thread(upsert_chunks, batch, embeddings, target_collection)
            batch_elapsed = time.time() - batch_t0

            total_upserted += len(batch)

            yield _sse_event("progress", {
                "phase": "embedding",
                "completed": total_upserted,
                "total": total_chunks,
                "elapsed_sec": round(time.time() - t0, 1),
                "batch_time_sec": round(batch_elapsed, 2),
            })

        elapsed = time.time() - t0
        chunks_per_sec = total_upserted / elapsed if elapsed > 0 else 0

        yield _sse_event("summary", {
            "embedding_model": embedding_model,
            "dimensions": model_info.dimensions,
            "files_processed": len(all_files),
            "chunks_ingested": total_upserted,
            "ingestion_time_sec": round(elapsed, 2),
            "chunks_per_sec": round(chunks_per_sec, 1),
        })
