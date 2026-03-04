import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from app.config import get_settings
from app.schemas import (
    QueryRequest, ChunkDetail, TimingDetail, TokenUsage,
    RetrievalDetails, QueryResponse, CapabilityRequest, TrialRequest,
    ExpandRequest, ExpandResponse,
)
from app.services.retrieval import retrieve, _expand_query, _extract_routine_name
from app.services.generation import generate_answer, generate_answer_stream
from app.services.capabilities import CAPABILITIES
from app.services.trial_store import save_trial, list_trials, delete_trial
from app.services.eval_runner import eval_stream_generator, e2e_eval_stream_generator
from app.services.ingest_runner import ingest_stream_generator
from app.models_data import MODELS
from app.logging_config import setup_logging
from app.sse import sse_event as _sse_event

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# --- Response cache for repeated queries (TTL-based, max 64 entries) ---
_RESPONSE_CACHE: OrderedDict[str, tuple[float, "QueryResponse"]] = OrderedDict()
_CACHE_MAX = 64
_CACHE_TTL = 300  # 5 minutes


def _cache_key(query: str, top_k: int, capability: str | None, model: str | None = None,
               expanded_names: list[str] | None = None) -> str:
    names_part = ",".join(sorted(expanded_names)) if expanded_names else ""
    raw = f"{query}|{top_k}|{capability}|{model}|{names_part}"
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


async def _build_response(query: str, top_k: int, capability: str | None = None, model: str | None = None, expanded_names: list[str] | None = None) -> QueryResponse:
    """Shared logic for query and capability endpoints."""
    key = _cache_key(query, top_k, capability, model, expanded_names)
    cached = _cache_get(key)
    if cached is not None:
        logger.info("Cache hit for query: %.80s", query)
        return cached

    t0 = time.time()

    try:
        retrieval_result = await retrieve(query, top_k, model=model, capability=capability, expanded_names=expanded_names)
    except Exception as e:
        logger.error("Retrieval failed for query '%.80s': %s", query, e)
        raise
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
    return await _build_response(req.query, req.top_k, None, model=req.model, expanded_names=req.expanded_names)


@app.post("/api/capabilities/{capability}", response_model=QueryResponse)
async def capability_endpoint(capability: str, req: CapabilityRequest):
    if capability not in CAPABILITIES:
        raise HTTPException(status_code=404, detail=f"Unknown capability: {capability}")
    return await _build_response(req.query, req.top_k, capability, model=req.model, expanded_names=req.expanded_names)


@app.post("/api/expand", response_model=ExpandResponse)
async def expand_endpoint(req: ExpandRequest):
    """Pre-expand a conceptual query into LAPACK routine names (for prefetch)."""
    query_hash = hashlib.md5(req.query.encode()).hexdigest()
    if _extract_routine_name(req.query):
        return ExpandResponse(expanded_names=[], query_hash=query_hash)
    names = await _expand_query(req.query, model=req.model)
    return ExpandResponse(expanded_names=names, query_hash=query_hash)


def _chunk_detail_to_dict(detail: ChunkDetail, text: str = "") -> dict:
    """Convert a ChunkDetail to dict, optionally including text for streaming."""
    d = detail.model_dump()
    if text:
        d["text"] = text
    return d


async def _stream_generator(query: str, top_k: int, capability: str | None = None, model: str | None = None, expanded_names: list[str] | None = None):
    """Async generator that yields SSE events: retrieval, token*, done."""
    t0 = time.time()

    try:
        retrieval_result = await retrieve(query, top_k, model=model, capability=capability, expanded_names=expanded_names)
    except Exception as e:
        logger.error("Stream retrieval failed: %s", e)
        yield _sse_event("error", {"message": f"Retrieval failed: {e}"})
        return
    chunks = retrieval_result["chunks"]
    t_retrieval = time.time()
    retrieval_ms = round((t_retrieval - t0) * 1000, 1)

    # Reuse _build_chunk_details, then add text for streaming
    chunk_details = _build_chunk_details(chunks)
    chunk_dicts = [
        _chunk_detail_to_dict(detail, text=chunk.get("text", ""))
        for detail, chunk in zip(chunk_details, chunks)
    ]

    yield _sse_event("retrieval", {
        "retrieval_details": {
            "strategy": retrieval_result["retrieval_strategy"],
            "expanded_names": retrieval_result["expanded_names"],
            "chunks": chunk_dicts,
        },
        "timing": {"retrieval_ms": retrieval_ms},
    })

    # Stream generation tokens
    try:
        async for event in generate_answer_stream(query, chunks, capability, model=model):
            if event["type"] == "token":
                yield _sse_event("token", {"token": event["token"]})
            elif event["type"] == "error":
                logger.error("Stream generation error: %s", event.get("message", "unknown"))
                yield _sse_event("error", {"message": event.get("message", "Generation failed")})
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
    except Exception as e:
        logger.error("Streaming error: %s", e)
        yield _sse_event("error", {"message": str(e)})


@app.post("/api/query/stream")
async def query_stream_endpoint(req: QueryRequest):
    return StreamingResponse(
        _stream_generator(req.query, req.top_k, None, model=req.model, expanded_names=req.expanded_names),
        media_type="text/event-stream",
    )


@app.post("/api/capabilities/{capability}/stream")
async def capability_stream_endpoint(capability: str, req: CapabilityRequest):
    if capability not in CAPABILITIES:
        raise HTTPException(status_code=404, detail=f"Unknown capability: {capability}")

    return StreamingResponse(
        _stream_generator(req.query, req.top_k, capability, model=req.model, expanded_names=req.expanded_names),
        media_type="text/event-stream",
    )


# --- Eval endpoints ---

@app.get("/api/eval/stream")
async def eval_stream_endpoint(model: str | None = None, embedding_model: str | None = None):
    return StreamingResponse(
        eval_stream_generator(model=model, embedding_model=embedding_model),
        media_type="text/event-stream",
    )


@app.get("/api/eval/e2e/stream")
async def e2e_eval_stream_endpoint(model: str | None = None):
    return StreamingResponse(
        e2e_eval_stream_generator(model=model), media_type="text/event-stream",
    )


@app.get("/api/ingest/stream")
async def ingest_stream_endpoint(embedding_model: str):
    return StreamingResponse(
        ingest_stream_generator(embedding_model=embedding_model),
        media_type="text/event-stream",
    )


# --- Model & Trial endpoints ---

@app.get("/api/embedding-models")
async def embedding_models_endpoint():
    from app.embedding_registry import EMBEDDING_MODELS
    return {
        "models": [
            {"name": info.name, "dimensions": info.dimensions, "provider": info.provider}
            for info in EMBEDDING_MODELS.values()
        ]
    }


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
        "avg_precision_at_5": req.avg_precision_at_5,
        "pass_rate": req.pass_rate,
        "avg_retrieval_latency_ms": req.avg_retrieval_latency_ms,
        "avg_e2e_latency_ms": req.avg_e2e_latency_ms,
        "total_queries": req.total_queries,
        "input_cost_per_1m": pricing.get("input_cost_per_1m"),
        "output_cost_per_1m": pricing.get("output_cost_per_1m"),
        "embedding_model": req.embedding_model,
        "embedding_dimensions": req.embedding_dimensions,
        "ingestion_time_sec": req.ingestion_time_sec,
        "chunks_ingested": req.chunks_ingested,
        "files_processed": req.files_processed,
        "notes": req.notes,
    }
    trial_id = save_trial(data)
    return {"id": trial_id}


@app.get("/api/trials")
async def list_trials_endpoint(eval_type: str | None = None):
    trials = list_trials()
    if eval_type:
        trials = [t for t in trials if t.get("eval_type") == eval_type]
    return {"trials": trials}


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
