"""Load pre-parsed chunks from gzipped JSONL fixture."""

import gzip
import json
import logging
from pathlib import Path

from app.services.chunker import Chunk

logger = logging.getLogger(__name__)

DEFAULT_FIXTURE_PATH = Path("fixtures/chunks.jsonl.gz")


def load_chunks_from_fixture(path: Path = DEFAULT_FIXTURE_PATH) -> list[Chunk]:
    """Load pre-parsed chunks from gzipped JSONL fixture.

    Each line is JSON: {"text": "...", "metadata": {...}}
    """
    chunks = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(Chunk(text=obj["text"], metadata=obj["metadata"]))
    logger.info("Loaded %d chunks from fixture: %s", len(chunks), path)
    return chunks
