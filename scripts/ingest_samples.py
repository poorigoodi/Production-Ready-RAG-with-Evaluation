#!/usr/bin/env python3
"""Ingest bundled sample documents into the local index."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.logging_utils import setup_logging
from app.services import init_services


def main() -> None:
    settings = get_settings()
    # Prefer mock for offline sample bootstrap unless explicitly disabled.
    if "--real-models" not in sys.argv:
        settings.mock_models = True
    setup_logging(settings.log_level)
    svc = init_services(settings)
    if not svc.ready:
        raise SystemExit(svc.init_error)

    sample_dir = ROOT / "data" / "documents" / "sample"
    files = sorted(
        list(sample_dir.glob("*.txt"))
        + list(sample_dir.glob("*.pdf"))
        + list(sample_dir.glob("*.docx"))
    )
    if not files:
        raise SystemExit(f"No sample files in {sample_dir}")

    id_map = {
        "company_policy.txt": "doc_001",
        "support_guide.txt": "doc_002",
        "hr_handbook.txt": "doc_003",
        "product_faq.txt": "doc_004",
        "persian_faq.txt": "doc_005",
        "company_policy.pdf": "doc_001",
        "support_guide.docx": "doc_002",
    }
    for path in files:
        doc_id = id_map.get(path.name)
        result = svc.ingestor.ingest_bytes(
            path.name, path.read_bytes(), document_id=doc_id
        )
        print(
            path.name,
            result["status"],
            result["document_id"],
            "dup=" + str(result["duplicate"]),
        )


if __name__ == "__main__":
    main()
