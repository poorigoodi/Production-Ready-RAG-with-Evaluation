"""Tests for validation, extraction, and chunking."""

from __future__ import annotations

import pytest

from app.chunking import chunk_document_text, split_fixed, split_paragraph_aware
from app.ingestion import IngestionError, extract_txt, normalize_text, validate_upload


def test_validate_upload_rejects_bad_extension():
    with pytest.raises(IngestionError):
        validate_upload("notes.exe", b"data", 20 * 1024 * 1024)


def test_validate_upload_rejects_oversized():
    with pytest.raises(IngestionError):
        validate_upload("a.txt", b"x" * 100, max_bytes=10)


def test_validate_upload_accepts_txt():
    ext = validate_upload("policy.txt", b"hello world", 20 * 1024 * 1024)
    assert ext == ".txt"


def test_extract_txt_utf8():
    blocks, pages = extract_txt("Hello café".encode("utf-8"))
    assert pages is None
    assert "café" in blocks[0].text


def test_normalize_persian_variants():
    # Arabic yeh/kaf -> Persian
    text = normalize_text("يك")
    assert "ی" in text or "\u06cc" in text
    assert "ک" in text or "\u06a9" in text


def test_fixed_chunking_overlap():
    text = "A" * 1000
    chunks = split_fixed(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 3
    assert all(len(c) <= 300 for c in chunks)


def test_paragraph_aware_chunking():
    text = "Para one.\n\nPara two is here.\n\nPara three finishes."
    chunks = split_paragraph_aware(text, chunk_size=40, chunk_overlap=5)
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)


def test_chunk_ids_stable():
    chunks = chunk_document_text(
        "Alpha paragraph.\n\nBeta paragraph with more text.",
        "doc_001",
        chunk_size=800,
        chunk_overlap=150,
        strategy="paragraph",
    )
    assert chunks[0].chunk_id == "doc_001_chunk_000"


def test_ingest_and_duplicate(app_services, sample_text):
    first = app_services.ingestor.ingest_bytes("policy.txt", sample_text, document_id="doc_test")
    assert first["status"] == "indexed"
    assert first["chunk_count"] >= 1
    second = app_services.ingestor.ingest_bytes("policy_copy.txt", sample_text)
    assert second["duplicate"] is True


def test_delete_document_clears_index(app_services, sample_text):
    out = app_services.ingestor.ingest_bytes("policy.txt", sample_text, document_id="doc_del")
    assert app_services.index_manager.faiss_store.size >= 1
    ok = app_services.ingestor.delete_document(out["document_id"])
    assert ok
    assert app_services.db.get_document(out["document_id"]) is None
    assert app_services.index_manager.faiss_store.size == 0
