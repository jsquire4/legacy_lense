import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.retrieval import retrieve
from app.services.generation import generate_answer
from app.services.capabilities import CAPABILITIES
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("LegacyLens started")
    yield


app = FastAPI(title="LegacyLens", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response models ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)


class ChunkDetail(BaseModel):
    rank: int
    chunk_id: str
    file_name: str
    routine_name: str
    score: float
    match_type: str


class TimingDetail(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RetrievalDetails(BaseModel):
    strategy: str
    expanded_names: list[str] = []
    chunks: list[ChunkDetail] = []


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    latency_ms: float
    retrieval_details: RetrievalDetails | None = None
    token_usage: TokenUsage | None = None
    timing: TimingDetail | None = None


class CapabilityRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


def _build_response(query: str, top_k: int, capability: str | None = None) -> QueryResponse:
    """Shared logic for query and capability endpoints."""
    t0 = time.time()

    retrieval_result = retrieve(query, top_k)
    chunks = retrieval_result["chunks"]
    t_retrieval = time.time()

    result = generate_answer(query, chunks, capability)
    t_generation = time.time()

    retrieval_ms = round((t_retrieval - t0) * 1000, 1)
    generation_ms = round((t_generation - t_retrieval) * 1000, 1)
    total_ms = round((t_generation - t0) * 1000, 1)

    # Build chunk details for observability
    chunk_details = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        file_path = meta.get("file_path", "")
        chunk_details.append(ChunkDetail(
            rank=i + 1,
            chunk_id=str(chunk.get("id", "")),
            file_name=Path(file_path).name if file_path else "",
            routine_name=meta.get("unit_name", ""),
            score=round(chunk.get("score", 0.0), 4),
            match_type=chunk.get("_match_type", "vector"),
        ))

    token_usage_raw = result.get("token_usage", {})
    token_usage = TokenUsage(
        prompt_tokens=token_usage_raw.get("prompt_tokens", 0),
        completion_tokens=token_usage_raw.get("completion_tokens", 0),
        total_tokens=token_usage_raw.get("total_tokens", 0),
    )

    # Structured log
    logger.info(
        "Query processed",
        extra={
            "query": query[:200],
            "chunk_ids": [c.chunk_id for c in chunk_details],
            "scores": [c.score for c in chunk_details],
            "latency_ms": total_ms,
            "token_usage": token_usage_raw,
        },
    )

    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        latency_ms=total_ms,
        retrieval_details=RetrievalDetails(
            strategy=retrieval_result["retrieval_strategy"],
            expanded_names=retrieval_result["expanded_names"],
            chunks=chunk_details,
        ),
        token_usage=token_usage,
        timing=TimingDetail(
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        ),
    )


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _build_response, req.query, req.top_k, None)


@app.post("/api/capabilities/{capability}", response_model=QueryResponse)
async def capability_endpoint(capability: str, req: CapabilityRequest):
    if capability not in CAPABILITIES:
        raise HTTPException(status_code=404, detail=f"Unknown capability: {capability}")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _build_response, req.query, req.top_k, capability)


@app.get("/")
async def root():
    return FileResponse(str(_STATIC_DIR / "index.html"))


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
