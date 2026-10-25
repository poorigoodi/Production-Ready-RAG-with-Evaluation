#!/usr/bin/env python3
"""Experiment C: compare chunk sizes 400 / 800 / 1200 with constant overlap ratio."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.database import reset_db_singleton
from app.embeddings import reset_embedder
from app.evaluation import run_pipeline_evaluation
from app.generation import reset_llm
from app.logging_utils import setup_logging
from app.reranking import reset_reranker
from app.services import init_services, reset_services

SAMPLE_DIR = ROOT / "data" / "documents" / "sample"
OVERLAP_RATIO = 150 / 800  # keep ratio consistent with defaults


def reset_all() -> None:
    get_settings.cache_clear()
    reset_db_singleton()
    reset_embedder()
    reset_reranker()
    reset_llm()
    reset_services()


def main() -> None:
    base = get_settings()
    base.mock_models = "--real-models" not in sys.argv
    setup_logging(base.log_level)

    summaries = []
    for chunk_size in (400, 800, 1200):
        reset_all()
        settings = get_settings()
        settings.mock_models = base.mock_models
        settings.chunk_size = chunk_size
        settings.chunk_overlap = max(1, int(chunk_size * OVERLAP_RATIO))
        # Isolate DB/index per configuration
        run_dir = ROOT / "data" / f"exp_chunk_{chunk_size}"
        settings.database_url = f"sqlite:///{(run_dir / 'rag.db').as_posix()}"
        settings.documents_dir = run_dir / "documents"
        settings.index_dir = run_dir / "indexes"
        settings.ensure_directories()

        svc = init_services(settings)
        id_map = {
            "company_policy.txt": "doc_001",
            "support_guide.txt": "doc_002",
            "hr_handbook.txt": "doc_003",
            "product_faq.txt": "doc_004",
            "persian_faq.txt": "doc_005",
        }
        for name, doc_id in id_map.items():
            path = SAMPLE_DIR / name
            svc.ingestor.ingest_bytes(path.name, path.read_bytes(), document_id=doc_id)

        report = run_pipeline_evaluation(
            svc.pipeline,
            modes=["hybrid"],
            top_k=5,
            experiment_name=f"chunk_{chunk_size}",
            settings=settings,
        )
        summaries.append(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "metrics": report["retrieval_metrics"],
                "report": report["artifacts"]["json"],
            }
        )
        print(json.dumps(summaries[-1], indent=2))

    out = ROOT / "reports" / "metrics" / "chunk_size_experiment_summary.json"
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
