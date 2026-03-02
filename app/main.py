import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.retrieval import retrieve
from app.services.generation import generate_answer, generate_answer_stream
from app.services.capabilities import CAPABILITIES
from app.logging_config import setup_logging
from app.eval_data import EVAL_QUERIES, E2E_EVAL_QUERIES, compute_recall_at_k, check_e2e_result

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


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event."""
    import json
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_generator(query: str, top_k: int, capability: str | None = None):
    """Generator that yields SSE events: retrieval, token*, done."""
    import json
    t0 = time.time()

    retrieval_result = retrieve(query, top_k)
    chunks = retrieval_result["chunks"]
    t_retrieval = time.time()
    retrieval_ms = round((t_retrieval - t0) * 1000, 1)

    # Build chunk details
    chunk_details = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        file_path = meta.get("file_path", "")
        chunk_details.append({
            "rank": i + 1,
            "chunk_id": str(chunk.get("id", "")),
            "file_name": Path(file_path).name if file_path else "",
            "routine_name": meta.get("unit_name", ""),
            "score": round(chunk.get("score", 0.0), 4),
            "match_type": chunk.get("_match_type", "vector"),
        })

    yield _sse_event("retrieval", {
        "retrieval_details": {
            "strategy": retrieval_result["retrieval_strategy"],
            "expanded_names": retrieval_result["expanded_names"],
            "chunks": chunk_details,
        },
        "timing": {"retrieval_ms": retrieval_ms},
    })

    # Stream generation tokens
    for event in generate_answer_stream(query, chunks, capability):
        if event["type"] == "token":
            yield _sse_event("token", {"token": event["token"]})
        elif event["type"] == "done":
            t_done = time.time()
            generation_ms = round((t_done - t_retrieval) * 1000, 1)
            total_ms = round((t_done - t0) * 1000, 1)
            yield _sse_event("done", {
                "citations": event["citations"],
                "token_usage": event["token_usage"],
                "timing": {
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": generation_ms,
                    "total_ms": total_ms,
                },
            })


@app.post("/api/query/stream")
async def query_stream_endpoint(req: QueryRequest):
    loop = asyncio.get_event_loop()

    def gen():
        yield from _stream_generator(req.query, req.top_k, None)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/capabilities/{capability}/stream")
async def capability_stream_endpoint(capability: str, req: CapabilityRequest):
    if capability not in CAPABILITIES:
        raise HTTPException(status_code=404, detail=f"Unknown capability: {capability}")

    def gen():
        yield from _stream_generator(req.query, req.top_k, capability)

    return StreamingResponse(gen(), media_type="text/event-stream")


def _eval_stream_generator():
    """Generator yielding SSE events for each retrieval eval query: progress*, summary."""
    total_recall = 0.0
    total_latency = 0.0
    n = len(EVAL_QUERIES)

    for i, item in enumerate(EVAL_QUERIES):
        query = item["query"]
        expected = item["expected_files"]

        t0 = time.time()
        result = retrieve(query, top_k=5)
        latency_ms = round((time.time() - t0) * 1000, 1)

        retrieved_files = []
        seen = set()
        for chunk in result["chunks"]:
            fp = chunk.get("metadata", {}).get("file_path", "")
            if fp:
                fname = Path(fp).name
                if fname not in seen:
                    retrieved_files.append(fname)
                    seen.add(fname)

        recall = compute_recall_at_k(retrieved_files, expected, k=5)
        total_recall += recall
        total_latency += latency_ms

        yield _sse_event("progress", {
            "index": i,
            "query": query,
            "capability": item.get("capability"),
            "recall_at_5": round(recall, 4),
            "latency_ms": latency_ms,
            "retrieved_files": retrieved_files[:5],
            "expected_files": expected,
        })

    yield _sse_event("summary", {
        "avg_recall_at_5": round(total_recall / n, 4),
        "avg_latency_ms": round(total_latency / n, 1),
        "total_queries": n,
    })


def _e2e_eval_stream_generator():
    """Generator yielding SSE events for e2e generation evals: progress*, summary."""
    n = len(E2E_EVAL_QUERIES)
    total_passed = 0
    total_latency = 0.0

    for i, item in enumerate(E2E_EVAL_QUERIES):
        query = item["query"]
        capability = item.get("capability")
        checks = item["checks"]

        t0 = time.time()
        retrieval_result = retrieve(query, top_k=8)
        chunks = retrieval_result["chunks"]
        gen_result = generate_answer(query, chunks, capability)
        latency_ms = round((time.time() - t0) * 1000, 1)

        check_results = check_e2e_result(
            gen_result["answer"], gen_result["citations"], checks,
        )

        if check_results["pass"]:
            total_passed += 1
        total_latency += latency_ms

        yield _sse_event("progress", {
            "index": i,
            "query": query,
            "capability": capability,
            "passed": check_results["pass"],
            "checks": check_results,
            "latency_ms": latency_ms,
            "citations_count": len(gen_result["citations"]),
            "answer_length": len(gen_result["answer"]),
        })

    yield _sse_event("summary", {
        "total_queries": n,
        "passed": total_passed,
        "failed": n - total_passed,
        "pass_rate": round(total_passed / n, 4) if n else 0,
        "avg_latency_ms": round(total_latency / n, 1) if n else 0,
    })


@app.get("/api/eval/stream")
async def eval_stream_endpoint():
    def gen():
        yield from _eval_stream_generator()
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/eval/e2e/stream")
async def e2e_eval_stream_endpoint():
    def gen():
        yield from _e2e_eval_stream_generator()
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
async def root():
    return FileResponse(str(_STATIC_DIR / "index.html"))


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
