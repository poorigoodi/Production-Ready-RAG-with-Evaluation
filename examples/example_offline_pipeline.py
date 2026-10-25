#!/usr/bin/env python3
"""Offline pipeline example (no HTTP server required).

Uses mock models by default so it runs without downloading weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.services import init_services


def main() -> None:
    settings = get_settings()
    settings.mock_models = True
    svc = init_services(settings)

    sample = ROOT / "data" / "documents" / "sample" / "company_policy.txt"
    ingested = svc.ingestor.ingest_bytes(
        sample.name,
        sample.read_bytes(),
        document_id="doc_example_policy",
    )
    print("Ingest:", json.dumps(ingested, indent=2))

    search = svc.pipeline.search(
        "refund period",
        retrieval_mode="hybrid",
        top_k=3,
        rerank=True,
    )
    print("\nTop chunks:")
    for rank, chunk in enumerate(search["results"], start=1):
        print(f"  {rank}. {chunk.chunk_id} score={chunk.score:.4f}")
        print(f"     {chunk.text[:120]}…")

    chat = svc.pipeline.chat("What is the refund policy?")
    print("\nAnswer:", chat["answer"])
    print("Sources:", json.dumps(chat["sources"], indent=2))
    print(f"Latency: {chat['latency_ms']:.1f} ms")


if __name__ == "__main__":
    main()
