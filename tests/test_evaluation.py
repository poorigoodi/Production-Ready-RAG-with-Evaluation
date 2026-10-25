"""Tests for retrieval/answer metrics using synthetic judgments."""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation import (
    EvaluationRunner,
    exact_match,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    resolve_relevance_labels,
    token_f1,
)


def test_recall_precision_mrr_ndcg():
    relevant = {"a", "b"}
    retrieved = ["x", "a", "b", "y"]
    assert recall_at_k(relevant, retrieved, 3) == 1.0
    assert precision_at_k(relevant, retrieved, 2) == 0.5
    assert mrr(relevant, retrieved) == 0.5
    assert ndcg_at_k(relevant, retrieved, 4) > 0


def test_exact_match_and_f1():
    assert exact_match("14 days", "14 days") == 1.0
    assert token_f1("refund in 14 days", "14 days") > 0


def test_resolve_relevance_spans():
    benchmark = {
        "version": "1.0.0",
        "items": [
            {
                "id": "e1",
                "question": "refund?",
                "relevant_spans": ["within 14 days"],
                "relevant_chunk_ids": [],
            }
        ],
    }
    chunks = [
        {"chunk_id": "doc_001_chunk_000", "text": "Refund requests are accepted within 14 days."},
        {"chunk_id": "doc_001_chunk_001", "text": "Unrelated warranty text."},
    ]
    resolved = resolve_relevance_labels(benchmark, chunks)
    assert resolved["items"][0]["relevant_chunk_ids"] == ["doc_001_chunk_000"]


def test_evaluation_runner_table(tmp_settings, app_services, sample_text):
    app_services.ingestor.ingest_bytes("policy.txt", sample_text, document_id="doc_001")
    chunks = app_services.db.list_chunks()
    refund_chunk = next(c["chunk_id"] for c in chunks if "14 days" in c["text"])
    benchmark = {
        "name": "synth",
        "version": "test",
        "items": [
            {
                "id": "e1",
                "question": "What is the refund period?",
                "reference_answer": "14 days",
                "relevant_chunk_ids": [refund_chunk],
            },
            {
                "id": "e2",
                "question": "What is the stock price?",
                "reference_answer": "unknown",
                "relevant_chunk_ids": [],
                "insufficient_evidence": True,
            },
        ],
    }

    def retrieve_fn(question, mode, top_k, do_rerank):
        out = app_services.pipeline.search(
            question, retrieval_mode=mode, top_k=top_k, rerank=do_rerank
        )
        return [r.chunk_id for r in out["results"]]

    runner = EvaluationRunner(tmp_settings)
    results = runner.evaluate_retrieval(
        benchmark,
        retrieve_fn,
        modes=[("Hybrid", "hybrid", False)],
        top_k=3,
    )
    assert results["sample_count"] == 1
    assert "Recall@3" in results["table"].columns
    report = runner.write_report(
        retrieval_results=results,
        answer_results=None,
        benchmark=benchmark,
        experiment_name="unit_test",
    )
    report_path = Path(report["artifacts"]["json"])
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["sample_count"] == 1
