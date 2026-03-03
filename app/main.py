import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
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
from app.services.trial_store import save_trial, list_trials, delete_trial
from app.models_data import MODELS, MODEL_NAMES
from app.logging_config import setup_logging
from app.eval_data import EVAL_QUERIES, E2E_EVAL_QUERIES, compute_recall_at_k, check_e2e_result

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# --- Response cache for repeated queries (TTL-based, max 64 entries) ---
_RESPONSE_CACHE: OrderedDict[str, tuple[float, "QueryResponse"]] = OrderedDict()
_CACHE_MAX = 64
_CACHE_TTL = 300  # 5 minutes


def _cache_key(query: str, top_k: int, capability: str | None, model: str | None = None) -> str:
    raw = f"{query}|{top_k}|{capability}|{model}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str):
    if key in _RESPONSE_CACHE:
        ts, resp = _RESPONSE_CACHE[key]
        if time.time() - ts < _CACHE_TTL:
            _RESPONSE_CACHE.move_to_end(key)
            return resp
        del _RESPONSE_CACHE[key]
    return None


def _cache_put(key: str, resp):
    _RESPONSE_CACHE[key] = (time.time(), resp)
    if len(_RESPONSE_CACHE) > _CACHE_MAX:
        _RESPONSE_CACHE.popitem(last=False)


_WARMUP_QUERIES = [
    ("What does the DGESV subroutine do?", None),
    ("Explain the DGETRF factorization routine", "explain_code"),
    ("Generate documentation for ZGEMM", "generate_docs"),
    ("What patterns are used in the BLAS routines?", "detect_patterns"),
    ("What does DGESV call internally?", "map_dependencies"),
    ("What breaks if DTRSM changes its interface?", "impact_analysis"),
    ("What numerical checks does DGESVD enforce?", "extract_business_rules"),
]


_WARMUP_MODELS = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-5-nano"]


async def _warm_cache():
    """Pre-cache responses for the default example queries across cheap models."""
    for model in _WARMUP_MODELS:
        for query, capability in _WARMUP_QUERIES:
            try:
                await _build_response(query, 5, capability, model)
                logger.info("Cache warmed [%s]: %.60s", model, query)
            except Exception as e:
                logger.warning("Cache warmup failed [%s] '%.60s': %s", model, query, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("LegacyLens started")
    asyncio.create_task(_warm_cache())
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
    model: str | None = None


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
    model: str | None = None


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


def _build_chunk_details(chunks: list[dict]) -> list[ChunkDetail]:
    """Build chunk details for observability."""
    details = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        file_path = meta.get("file_path", "")
        details.append(ChunkDetail(
            rank=i + 1,
            chunk_id=str(chunk.get("id", "")),
            file_name=Path(file_path).name if file_path else "",
            routine_name=meta.get("unit_name", ""),
            score=round(chunk.get("score", 0.0), 4),
            match_type=chunk.get("_match_type", "vector"),
        ))
    return details


async def _build_response(query: str, top_k: int, capability: str | None = None, model: str | None = None) -> QueryResponse:
    """Shared logic for query and capability endpoints."""
    key = _cache_key(query, top_k, capability, model)
    cached = _cache_get(key)
    if cached is not None:
        logger.info("Cache hit for query: %.80s", query)
        return cached

    t0 = time.time()

    retrieval_result = await retrieve(query, top_k, model=model)
    chunks = retrieval_result["chunks"]
    t_retrieval = time.time()

    result = await generate_answer(query, chunks, capability, model=model)
    t_generation = time.time()

    retrieval_ms = round((t_retrieval - t0) * 1000, 1)
    generation_ms = round((t_generation - t_retrieval) * 1000, 1)
    total_ms = round((t_generation - t0) * 1000, 1)

    chunk_details = _build_chunk_details(chunks)

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

    response = QueryResponse(
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
    _cache_put(key, response)
    return response


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    return await _build_response(req.query, req.top_k, None, model=req.model)


@app.post("/api/capabilities/{capability}", response_model=QueryResponse)
async def capability_endpoint(capability: str, req: CapabilityRequest):
    if capability not in CAPABILITIES:
        raise HTTPException(status_code=404, detail=f"Unknown capability: {capability}")
    return await _build_response(req.query, req.top_k, capability, model=req.model)


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_generator(query: str, top_k: int, capability: str | None = None, model: str | None = None):
    """Async generator that yields SSE events: retrieval, token*, done."""
    t0 = time.time()

    retrieval_result = await retrieve(query, top_k, model=model)
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
            "text": chunk.get("text", ""),
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
    async for event in generate_answer_stream(query, chunks, capability, model=model):
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
    return StreamingResponse(
        _stream_generator(req.query, req.top_k, None, model=req.model),
        media_type="text/event-stream",
    )


@app.post("/api/capabilities/{capability}/stream")
async def capability_stream_endpoint(capability: str, req: CapabilityRequest):
    if capability not in CAPABILITIES:
        raise HTTPException(status_code=404, detail=f"Unknown capability: {capability}")

    return StreamingResponse(
        _stream_generator(req.query, req.top_k, capability, model=req.model),
        media_type="text/event-stream",
    )


# --- Eval endpoints (async — uses same fast pipeline as query endpoints) ---

_RETRIEVAL_BATCH_SIZE = 5


async def _eval_stream_generator(model: str | None = None):
    """Async generator — batch retrieval evals (no LLM generation, safe to parallelize)."""
    total_recall = 0.0
    total_latency = 0.0
    n = len(EVAL_QUERIES)

    for batch_start in range(0, n, _RETRIEVAL_BATCH_SIZE):
        batch = list(enumerate(EVAL_QUERIES))[batch_start:batch_start + _RETRIEVAL_BATCH_SIZE]

        async def run_one(i, item):
            query = item["query"]
            expected = item["expected_files"]
            t0 = time.time()
            result = await retrieve(query, top_k=5, model=model)
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
            return {
                "index": i, "query": query,
                "capability": item.get("capability"),
                "recall_at_5": round(recall, 4), "latency_ms": latency_ms,
                "retrieved_files": retrieved_files[:5], "expected_files": expected,
            }

        results = await asyncio.gather(*[run_one(i, item) for i, item in batch])
        for r in sorted(results, key=lambda x: x["index"]):
            total_recall += r["recall_at_5"]
            total_latency += r["latency_ms"]
            yield _sse_event("progress", r)

    yield _sse_event("summary", {
        "avg_recall_at_5": round(total_recall / n, 4),
        "avg_latency_ms": round(total_latency / n, 1),
        "total_queries": n,
    })


_E2E_MAX_TOKENS = 2048
_E2E_CONTEXT_BUDGET = 3000


async def _e2e_eval_stream_generator(model: str | None = None):
    """Async generator — batch retrieval, then generate sequentially.

    Retrieval is fast (Qdrant + embed) and safe to fully parallelize.
    Generation runs one at a time with reduced token limits and a
    dedicated client (max_retries=1) to avoid hidden backoff delays.
    """
    n = len(E2E_EVAL_QUERIES)
    total_passed = 0
    total_latency = 0.0

    # Phase 1: batch all retrievals concurrently (fast)
    async def run_retrieval(i, item):
        return i, item, await retrieve(item["query"], top_k=5, model=model)

    retrieval_results = await asyncio.gather(*[
        run_retrieval(i, item) for i, item in enumerate(E2E_EVAL_QUERIES)
    ])

    # Phase 2: generate sequentially with reduced limits
    for i, item, retrieval_result in sorted(retrieval_results, key=lambda x: x[0]):
        query = item["query"]
        capability = item.get("capability")
        checks = item["checks"]
        chunks = retrieval_result["chunks"]

        t0 = time.time()
        gen_result = await generate_answer(
            query, chunks, capability,
            max_completion_tokens=_E2E_MAX_TOKENS,
            context_budget=_E2E_CONTEXT_BUDGET,
            model=model,
        )
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
            "citations": gen_result["citations"],
            "answer_preview": gen_result["answer"][:500],
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
async def eval_stream_endpoint(model: str | None = None):
    return StreamingResponse(
        _eval_stream_generator(model=model), media_type="text/event-stream",
    )


@app.get("/api/eval/e2e/stream")
async def e2e_eval_stream_endpoint(model: str | None = None):
    return StreamingResponse(
        _e2e_eval_stream_generator(model=model), media_type="text/event-stream",
    )


# --- Model & Trial endpoints ---

class TrialRequest(BaseModel):
    model: str
    eval_type: str
    avg_recall_at_5: float | None = None
    pass_rate: float | None = None
    avg_retrieval_latency_ms: float | None = None
    avg_e2e_latency_ms: float | None = None
    total_queries: int | None = None
    notes: str = ""


@app.get("/api/models")
async def models_endpoint():
    settings = get_settings()
    return {
        "models": [
            {"name": name, "default": name == settings.CHAT_MODEL, **info}
            for name, info in MODELS.items()
        ]
    }


@app.post("/api/trials")
async def create_trial_endpoint(req: TrialRequest):
    pricing = MODELS.get(req.model, {})
    data = {
        "model": req.model,
        "eval_type": req.eval_type,
        "avg_recall_at_5": req.avg_recall_at_5,
        "pass_rate": req.pass_rate,
        "avg_retrieval_latency_ms": req.avg_retrieval_latency_ms,
        "avg_e2e_latency_ms": req.avg_e2e_latency_ms,
        "total_queries": req.total_queries,
        "input_cost_per_1m": pricing.get("input_cost_per_1m"),
        "output_cost_per_1m": pricing.get("output_cost_per_1m"),
        "notes": req.notes,
    }
    trial_id = save_trial(data)
    return {"id": trial_id}


@app.get("/api/trials")
async def list_trials_endpoint():
    return {"trials": list_trials()}


@app.delete("/api/trials/{trial_id}")
async def delete_trial_endpoint(trial_id: int):
    deleted = delete_trial(trial_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Trial not found")
    return {"deleted": True}


@app.get("/")
async def root():
    return FileResponse(
        str(_STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-cache"},
    )


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
