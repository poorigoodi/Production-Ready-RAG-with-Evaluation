#!/usr/bin/env python3
"""Minimal end-to-end API examples against a running FastAPI server.

Start the API first:
    $env:MOCK_MODELS="true"
    uvicorn app.main:app --host 127.0.0.1 --port 8000

Then:
    python examples/example_api_usage.py
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

API = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "documents" / "sample" / "company_policy.txt"


def main() -> None:
    with httpx.Client(base_url=API, timeout=60.0) as client:
        health = client.get("/health").json()
        print("=== GET /health ===")
        print(json.dumps(health, indent=2))

        print("\n=== POST /documents/upload ===")
        with SAMPLE.open("rb") as f:
            upload = client.post(
                "/documents/upload",
                files={"file": ("company_policy.txt", f, "text/plain")},
            ).json()
        print(json.dumps(upload, indent=2))

        print("\n=== POST /search ===")
        search = client.post(
            "/search",
            json={
                "query": "What is the refund period?",
                "retrieval_mode": "hybrid",
                "top_k": 3,
                "rerank": True,
            },
        ).json()
        # Truncate long chunk text for readable console output
        for item in search.get("results", []):
            text = item.get("text", "")
            item["text"] = text[:160] + ("…" if len(text) > 160 else "")
        print(json.dumps(search, indent=2, ensure_ascii=False))

        print("\n=== POST /chat ===")
        chat = client.post(
            "/chat",
            json={
                "question": "What is the refund policy?",
                "retrieval_mode": "hybrid",
                "top_k": 3,
                "rerank": True,
            },
        ).json()
        print(json.dumps(chat, indent=2, ensure_ascii=False))

        print("\n=== Follow-up chat ===")
        followup = client.post(
            "/chat",
            json={
                "question": "Does it cover accidental damage?",
                "conversation_id": chat["conversation_id"],
                "retrieval_mode": "hybrid",
                "top_k": 3,
                "rerank": True,
            },
        ).json()
        print(json.dumps(followup, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
