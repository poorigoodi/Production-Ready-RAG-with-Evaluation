"""Tests for context construction, citations, and grounded generation."""

from __future__ import annotations

from app.generation import ABSTENTION_MESSAGE, GroundedGenerator, MockLLM, build_context
from app.retrieval import ScoredChunk


def _chunk(cid: str, text: str, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=cid,
        document_id="doc_001",
        document="policy.txt",
        page=1,
        text=text,
        score=score,
        dense_score=score,
    )


def test_build_context_includes_source_headers():
    ctx, selected = build_context(
        [_chunk("doc_001_chunk_000", "Refund requests are accepted within 14 days.")],
        max_tokens=512,
        reserve_tokens=64,
    )
    assert "[Source 1]" in ctx
    assert "policy.txt" in ctx
    assert selected[0].chunk_id == "doc_001_chunk_000"


def test_context_deduplicates_overlap():
    a = _chunk("c1", "Refund requests are accepted within 14 days of purchase.")
    b = _chunk("c2", "Refund requests are accepted within 14 days of purchase.")
    ctx, selected = build_context([a, b], max_tokens=512, reserve_tokens=64)
    assert len(selected) == 1


def test_grounded_answer_cites_source():
    gen = GroundedGenerator(MockLLM())
    answer, selected, abstained, _ = gen.generate_answer(
        "What is the refund policy?",
        [_chunk("c1", "Refund requests are accepted within 14 days of purchase.")],
    )
    assert not abstained
    assert "14 days" in answer
    assert "[Source 1]" in answer
    assert selected


def test_abstain_on_empty_retrieval():
    gen = GroundedGenerator(MockLLM())
    answer, selected, abstained, _ = gen.generate_answer("Anything?", [])
    assert abstained
    assert answer == ABSTENTION_MESSAGE
    assert selected == []


def test_chat_persists_history(app_services, sample_text):
    app_services.ingestor.ingest_bytes("policy.txt", sample_text, document_id="doc_001")
    first = app_services.pipeline.chat("What is the refund period?")
    assert first["conversation_id"]
    second = app_services.pipeline.chat(
        "Does it cover shipping fees?",
        conversation_id=first["conversation_id"],
    )
    msgs = app_services.db.get_recent_messages(first["conversation_id"], limit=10)
    assert len(msgs) >= 4
    assert second["conversation_id"] == first["conversation_id"]
