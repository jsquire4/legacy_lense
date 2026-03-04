"""Pydantic request/response models for the LegacyLens API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)
    model: str | None = None
    expanded_names: list[str] | None = None


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
    expanded_names: list[str] | None = None


class ExpandRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    model: str | None = None


class ExpandResponse(BaseModel):
    expanded_names: list[str] = []
    query_hash: str


class TrialRequest(BaseModel):
    model: str
    eval_type: str
    avg_recall_at_5: float | None = None
    avg_precision_at_5: float | None = None
    pass_rate: float | None = None
    avg_retrieval_latency_ms: float | None = None
    avg_e2e_latency_ms: float | None = None
    total_queries: int | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    notes: str = ""
