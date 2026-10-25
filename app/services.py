"""Process-wide service container for FastAPI / Streamlit / scripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.database import Database, get_db
from app.embeddings import Embedder, get_embedder
from app.generation import GroundedGenerator, get_llm
from app.ingestion import DocumentIngestor
from app.rag_pipeline import RAGPipeline
from app.reranking import get_reranker
from app.retrieval import IndexManager

logger = logging.getLogger(__name__)


@dataclass
class AppServices:
    settings: Settings
    db: Database
    embedder: Embedder
    index_manager: IndexManager
    ingestor: DocumentIngestor
    pipeline: RAGPipeline
    ready: bool = False
    init_error: str | None = None
    _extra: dict = field(default_factory=dict)


_services: AppServices | None = None


def init_services(settings: Settings | None = None) -> AppServices:
    global _services
    settings = settings or get_settings()
    settings.ensure_directories()
    db = get_db(settings)

    try:
        embedder = get_embedder(settings)
        index_manager = IndexManager(settings, db, embedder)
        index_manager.initialize()
        reranker = get_reranker(settings)
        llm = get_llm(settings)
        generator = GroundedGenerator(llm, settings)
        ingestor = DocumentIngestor(settings, db, embedder, index_manager)
        pipeline = RAGPipeline(settings, db)
        pipeline.bind_services(embedder, index_manager, reranker, generator)
        _services = AppServices(
            settings=settings,
            db=db,
            embedder=embedder,
            index_manager=index_manager,
            ingestor=ingestor,
            pipeline=pipeline,
            ready=True,
            init_error=None,
        )
        logger.info("Application services initialized (mock_models=%s)", settings.mock_models)
        return _services
    except Exception as exc:
        logger.exception("Service initialization failed")
        # Minimal services so health endpoint can report failure.
        embedder = get_embedder(settings) if settings.mock_models else None  # type: ignore
        _services = AppServices(
            settings=settings,
            db=db,
            embedder=embedder,  # type: ignore[arg-type]
            index_manager=IndexManager(settings, db, embedder),  # type: ignore[arg-type]
            ingestor=DocumentIngestor(settings, db),
            pipeline=RAGPipeline(settings, db),
            ready=False,
            init_error=str(exc),
        )
        return _services


def get_services() -> AppServices:
    global _services
    if _services is None:
        return init_services()
    return _services


def reset_services() -> None:
    global _services
    _services = None
