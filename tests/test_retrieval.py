"""Tests for BM25 tokenization, RRF, and hybrid retrieval."""

from __future__ import annotations

import numpy as np

from app.embeddings import MockEmbedder
from app.retrieval import (
    FaissIndexStore,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenize_english_and_persian():
    tokens = tokenize("Refund policy بازگشت کالا")
    assert "refund" in tokens
    assert any("بازگشت" in t or t == "بازگشت" for t in tokens)


def test_rrf_prefers_consensus():
    fused = reciprocal_rank_fusion(
        [["a", "b", "c"], ["b", "a", "d"]],
        k=60,
    )
    ranking = [cid for cid, _ in fused]
    assert ranking[0] in {"a", "b"}
    assert set(ranking) == {"a", "b", "c", "d"}


def test_faiss_cosine_via_ip(tmp_path):
    store = FaissIndexStore(tmp_path, dimension=8)
    embedder = MockEmbedder(dimension=8)
    texts = ["refund policy fourteen days", "unrelated astronomy facts"]
    vecs = embedder.embed_documents(texts)
    store.add(["c1", "c2"], vecs)
    q = embedder.embed_query("refund fourteen days")
    hits = store.search(q, top_k=2)
    assert hits[0][0] == "c1"


def test_hybrid_retrieval_returns_refund_chunk(app_services, sample_text):
    app_services.ingestor.ingest_bytes("policy.txt", sample_text, document_id="doc_001")
    results, latency = app_services.index_manager.retrieve(
        "What is the refund period?",
        mode="hybrid",
        top_k=3,
    )
    assert results
    assert "dense_ms" in latency
    joined = " ".join(r.text.lower() for r in results)
    assert "14 days" in joined or "refund" in joined


def test_metadata_filter(app_services, sample_text):
    app_services.ingestor.ingest_bytes("policy.txt", sample_text, document_id="doc_001")
    other = b"Totally different document about laptops and batteries only."
    app_services.ingestor.ingest_bytes("other.txt", other, document_id="doc_002")
    results, _ = app_services.index_manager.retrieve(
        "refund",
        mode="sparse",
        top_k=5,
        document_ids=["doc_001"],
    )
    assert all(r.document_id == "doc_001" for r in results)
