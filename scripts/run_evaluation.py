#!/usr/bin/env python3
"""CLI entrypoint for retrieval/answer evaluation experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.evaluation import run_pipeline_evaluation
from app.logging_utils import setup_logging
from app.services import init_services


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["dense", "sparse", "hybrid", "hybrid_rerank"],
    )
    parser.add_argument("--answers", action="store_true")
    parser.add_argument("--experiment", type=str, default="cli_eval")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if args.mock:
        settings.mock_models = True
    setup_logging(settings.log_level)
    svc = init_services(settings)
    if not svc.ready:
        raise SystemExit(f"Init failed: {svc.init_error}")

    report = run_pipeline_evaluation(
        svc.pipeline,
        benchmark_path=args.benchmark,
        modes=args.modes,
        top_k=args.top_k,
        include_answer_eval=args.answers,
        experiment_name=args.experiment,
        settings=settings,
    )
    print(json.dumps(report["retrieval_metrics"], indent=2))
    print(f"\nReport: {report['artifacts']['json']}")


if __name__ == "__main__":
    main()
