#!/usr/bin/env python3
"""Rebuild FAISS + BM25 indexes from SQLite chunk metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.logging_utils import setup_logging
from app.services import init_services


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild retrieval indexes")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock models (no weight downloads)",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.mock:
        settings.mock_models = True
    setup_logging(settings.log_level)
    svc = init_services(settings)
    if not svc.ready:
        raise SystemExit(f"Init failed: {svc.init_error}")
    svc.index_manager.rebuild_from_db()
    print(
        f"Index rebuilt. FAISS size={svc.index_manager.faiss_store.size if svc.index_manager.faiss_store else 0}"
    )


if __name__ == "__main__":
    main()
