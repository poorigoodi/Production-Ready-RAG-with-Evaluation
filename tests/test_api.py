"""API integration tests with mocked models."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_settings, monkeypatch):
    from app.services import init_services, reset_services

    reset_services()
    init_services(tmp_settings)

    from app.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["mock_models"] is True
    assert body["status"] in {"ok", "degraded"}


def test_upload_list_search_chat_delete(client, sample_text):
    files = {"file": ("policy.txt", sample_text, "text/plain")}
    r = client.post("/documents/upload", files=files)
    assert r.status_code == 200
    doc_id = r.json()["document_id"]

    r = client.get("/documents")
    assert r.status_code == 200
    assert r.json()["total"] >= 1

    r = client.post(
        "/search",
        json={"query": "refund period", "retrieval_mode": "hybrid", "top_k": 3},
    )
    assert r.status_code == 200
    assert isinstance(r.json()["results"], list)

    r = client.post(
        "/chat",
        json={"question": "What is the refund period?", "rerank": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "conversation_id" in body

    r = client.delete(f"/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True


def test_upload_rejects_bad_file(client):
    files = {"file": ("malware.exe", b"MZ123", "application/octet-stream")}
    r = client.post("/documents/upload", files=files)
    assert r.status_code == 400
