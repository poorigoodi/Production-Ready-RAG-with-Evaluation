"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    embedding_model: str
    llm_model: str
    index_ready: bool
    mock_models: bool


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_hash: str
    upload_timestamp: str
    page_count: int | None = None
    chunk_count: int = 0
    processing_status: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    page_count: int | None = None
    status: str
    duplicate: bool = False
    message: str | None = None


class SourceCitation(BaseModel):
    document: str
    page: int | None = None
    chunk_id: str
    score: float | None = None
    text_preview: str | None = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    conversation_id: str | None = None
    retrieval_mode: Literal["dense", "sparse", "hybrid"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool | None = None


class LatencyBreakdown(BaseModel):
    dense_ms: float | None = None
    bm25_ms: float | None = None
    fusion_ms: float | None = None
    rerank_ms: float | None = None
    generation_ms: float | None = None
    total_ms: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    latency_ms: float
    conversation_id: str
    abstained: bool = False
    rewritten_query: str | None = None
    latency_breakdown: LatencyBreakdown | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    retrieval_mode: Literal["dense", "sparse", "hybrid"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool | None = None
    document_ids: list[str] | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document: str
    page: int | None = None
    text: str
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
    latency_ms: float
    retrieval_mode: str
    reranked: bool


class EvaluationRunRequest(BaseModel):
    benchmark_path: str | None = None
    modes: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    include_answer_eval: bool = False
    experiment_name: str | None = None


class EvaluationRunResponse(BaseModel):
    report_id: str
    report_path: str
    metrics: dict[str, Any]
    sample_count: int


class EvaluationResultsResponse(BaseModel):
    reports: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    detail: str
    error_category: str | None = None
