import asyncio
import time
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

app = FastAPI(title="LegacyLens", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict before production deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    latency_ms: float


class CapabilityRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    start = time.time()
    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(None, retrieve, req.query, req.top_k)
    result = await loop.run_in_executor(None, generate_answer, req.query, chunks)
    latency_ms = (time.time() - start) * 1000
    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        latency_ms=round(latency_ms, 1),
    )


@app.post("/api/capabilities/{capability}", response_model=QueryResponse)
async def capability_endpoint(capability: str, req: CapabilityRequest):
    if capability not in CAPABILITIES:
        raise HTTPException(status_code=404, detail=f"Unknown capability: {capability}")
    start = time.time()
    loop = asyncio.get_event_loop()
    chunks = await loop.run_in_executor(None, retrieve, req.query, req.top_k)
    result = await loop.run_in_executor(
        None, generate_answer, req.query, chunks, capability
    )
    latency_ms = (time.time() - start) * 1000
    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        latency_ms=round(latency_ms, 1),
    )


@app.get("/")
async def root():
    return FileResponse(str(_STATIC_DIR / "index.html"))


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
