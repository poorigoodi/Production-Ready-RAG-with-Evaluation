"""Shared pytest fixtures using temporary directories and mock models."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Force mock models before app imports configure heavy backends.
os.environ["MOCK_MODELS"] = "true"
os.environ["APP_ENV"] = "test"


@pytest.fixture()
def tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.config import Settings, get_settings
    from app.database import reset_db_singleton
    from app.embeddings import reset_embedder
    from app.generation import reset_llm
    from app.reranking import reset_reranker
    from app.services import reset_services

    db_path = tmp_path / "test.db"
    docs = tmp_path / "documents"
    indexes = tmp_path / "indexes"
    evaluation = tmp_path / "evaluation"
    reports = tmp_path / "reports"
    for p in (docs, indexes, evaluation, reports / "metrics", reports / "figures"):
        p.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("MOCK_MODELS", "true")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("DOCUMENTS_DIR", str(docs))
    monkeypatch.setenv("INDEX_DIR", str(indexes))
    monkeypatch.setenv("EVALUATION_DIR", str(evaluation))
    monkeypatch.setenv("REPORTS_DIR", str(reports))
    monkeypatch.setenv("ENABLE_RERANKING", "true")

    get_settings.cache_clear()
    reset_db_singleton()
    reset_embedder()
    reset_reranker()
    reset_llm()
    reset_services()

    settings = Settings(
        mock_models=True,
        database_url=f"sqlite:///{db_path.as_posix()}",
        documents_dir=docs,
        index_dir=indexes,
        evaluation_dir=evaluation,
        reports_dir=reports,
        enable_reranking=True,
    )
    settings.ensure_directories()

    # Ensure cached settings match fixture
    get_settings.cache_clear()
    monkeypatch.setattr("app.config.get_settings", lambda: settings)

    yield settings

    get_settings.cache_clear()
    reset_db_singleton()
    reset_embedder()
    reset_reranker()
    reset_llm()
    reset_services()


@pytest.fixture()
def app_services(tmp_settings):
    from app.services import init_services

    return init_services(tmp_settings)


@pytest.fixture()
def sample_text() -> bytes:
    return (
        b"Refund Policy\n\n"
        b"Refund requests are accepted within 14 days of purchase. "
        b"Shipping fees are non-refundable unless caused by a fulfillment error.\n\n"
        b"Warranty Policy\n\n"
        b"The warranty period is 18 months from the delivery date. "
        b"Accidental damage is not covered.\n"
    )
