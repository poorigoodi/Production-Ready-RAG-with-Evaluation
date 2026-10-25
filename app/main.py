"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import get_settings
from app.evaluation import EvaluationRunner, run_pipeline_evaluation
from app.ingestion import IngestionError
from app.logging_utils import new_request_id, setup_logging
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    DocumentListResponse,
    EvaluationResultsResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    HealthResponse,
    LatencyBreakdown,
    RetrievedChunk,
    SearchRequest,
    SearchResponse,
    SourceCitation,
    UploadResponse,
)
from app.services import get_services, init_services

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    init_services(settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Production-oriented RAG API with hybrid retrieval, reranking, "
            "grounded generation, and evaluation. "
            "Not hardened for unauthenticated public internet exposure."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        svc = get_services()
        return HealthResponse(
            status="ok" if svc.ready else "degraded",
            app_name=svc.settings.app_name,
            version=__version__,
            embedding_model=svc.settings.embedding_model,
            llm_model=svc.settings.llm_model,
            index_ready=bool(svc.index_manager and svc.index_manager.ready),
            mock_models=svc.settings.mock_models,
        )

    @app.post("/documents/upload", response_model=UploadResponse)
    async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
        svc = get_services()
        if not svc.ready:
            raise HTTPException(
                status_code=503,
                detail=svc.init_error or "Services not ready",
            )
        data = await file.read()
        request_id = new_request_id()
        try:
            result = svc.ingestor.ingest_bytes(
                file.filename or "upload.bin",
                data,
                request_id=request_id,
            )
        except IngestionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Upload failed")
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

        return UploadResponse(**result)

    @app.get("/documents", response_model=DocumentListResponse)
    def list_documents() -> DocumentListResponse:
        svc = get_services()
        docs = [DocumentInfo(**d) for d in svc.db.list_documents()]
        return DocumentListResponse(documents=docs, total=len(docs))

    @app.delete("/documents/{document_id}")
    def delete_document(document_id: str) -> dict[str, Any]:
        svc = get_services()
        if not svc.ready:
            raise HTTPException(status_code=503, detail="Services not ready")
        ok = svc.ingestor.delete_document(document_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": True, "document_id": document_id}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        svc = get_services()
        if not svc.ready:
            raise HTTPException(status_code=503, detail=svc.init_error or "Services not ready")
        try:
            out = svc.pipeline.search(
                req.query,
                retrieval_mode=req.retrieval_mode,
                top_k=req.top_k,
                rerank=req.rerank,
                document_ids=req.document_ids,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        results = [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document=r.document,
                page=r.page,
                text=r.text,
                score=r.score,
                dense_score=r.dense_score,
                bm25_score=r.bm25_score,
                rerank_score=r.rerank_score,
            )
            for r in out["results"]
        ]
        return SearchResponse(
            query=out["query"],
            results=results,
            latency_ms=out["latency_ms"],
            retrieval_mode=out["retrieval_mode"],
            reranked=out["reranked"],
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        svc = get_services()
        if not svc.ready:
            raise HTTPException(status_code=503, detail=svc.init_error or "Services not ready")
        try:
            out = svc.pipeline.chat(
                req.question,
                conversation_id=req.conversation_id,
                retrieval_mode=req.retrieval_mode,
                top_k=req.top_k,
                rerank=req.rerank,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        breakdown = out.get("latency_breakdown")
        return ChatResponse(
            answer=out["answer"],
            sources=[SourceCitation(**s) for s in out["sources"]],
            latency_ms=out["latency_ms"],
            conversation_id=out["conversation_id"],
            abstained=out["abstained"],
            rewritten_query=out.get("rewritten_query"),
            latency_breakdown=LatencyBreakdown(**breakdown) if breakdown else None,
        )

    @app.delete("/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str) -> dict[str, Any]:
        svc = get_services()
        ok = svc.pipeline.clear_conversation(conversation_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"deleted": True, "conversation_id": conversation_id}

    @app.post("/evaluation/run", response_model=EvaluationRunResponse)
    def run_evaluation(req: EvaluationRunRequest) -> EvaluationRunResponse:
        svc = get_services()
        if not svc.ready:
            raise HTTPException(status_code=503, detail=svc.init_error or "Services not ready")
        bench = Path(req.benchmark_path) if req.benchmark_path else None
        try:
            report = run_pipeline_evaluation(
                svc.pipeline,
                benchmark_path=bench,
                modes=req.modes,
                top_k=req.top_k,
                include_answer_eval=req.include_answer_eval,
                experiment_name=req.experiment_name,
                settings=svc.settings,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Evaluation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return EvaluationRunResponse(
            report_id=report["report_id"],
            report_path=report["artifacts"]["json"],
            metrics={
                "retrieval": report["retrieval_metrics"],
                "answer": report.get("answer_metrics"),
            },
            sample_count=report["sample_count"],
        )

    @app.get("/evaluation/results", response_model=EvaluationResultsResponse)
    def evaluation_results() -> EvaluationResultsResponse:
        svc = get_services()
        runner = EvaluationRunner(svc.settings)
        return EvaluationResultsResponse(reports=runner.list_reports())

    return app


app = create_app()
