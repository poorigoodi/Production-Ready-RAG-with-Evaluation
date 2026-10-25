"""Cross-encoder reranking with optional disable and mock support."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Protocol

from app.config import Settings, get_settings
from app.retrieval import ScoredChunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    model_name: str
    enabled: bool

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> tuple[list[ScoredChunk], float]: ...


class NoOpReranker:
    model_name = "noop"
    enabled = False

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> tuple[list[ScoredChunk], float]:
        return candidates[:top_k], 0.0


class MockReranker:
    """Lexical overlap proxy for tests — not a substitute for cross-encoder quality."""

    def __init__(self, model_name: str = "mock-reranker") -> None:
        self.model_name = model_name
        self.enabled = True

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> tuple[list[ScoredChunk], float]:
        t0 = time.perf_counter()
        q_tokens = set(query.lower().split())
        scored: list[ScoredChunk] = []
        for c in candidates:
            overlap = len(q_tokens & set(c.text.lower().split()))
            # Stable tie-break
            bump = int(hashlib.md5(c.chunk_id.encode()).hexdigest()[:4], 16) / 1e6
            new = ScoredChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document=c.document,
                page=c.page,
                text=c.text,
                score=float(overlap) + bump,
                dense_score=c.dense_score,
                bm25_score=c.bm25_score,
                rerank_score=float(overlap) + bump,
            )
            scored.append(new)
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k], (time.perf_counter() - t0) * 1000


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        from sentence_transformers import CrossEncoder

        from app.embeddings import detect_device

        self.model_name = model_name
        self.enabled = True
        self.device = detect_device(device)
        logger.info("Loading reranker %s on %s", model_name, self.device)
        self._model = CrossEncoder(model_name, device=self.device)

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> tuple[list[ScoredChunk], float]:
        if not candidates:
            return [], 0.0
        t0 = time.perf_counter()
        pairs = [(query, c.text) for c in candidates]
        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception:
            logger.exception("Reranker failed; returning original order")
            return candidates[:top_k], (time.perf_counter() - t0) * 1000

        rescored: list[ScoredChunk] = []
        for c, s in zip(candidates, scores):
            rescored.append(
                ScoredChunk(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    document=c.document,
                    page=c.page,
                    text=c.text,
                    score=float(s),
                    dense_score=c.dense_score,
                    bm25_score=c.bm25_score,
                    rerank_score=float(s),
                )
            )
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k], (time.perf_counter() - t0) * 1000


_reranker_lock = threading.Lock()
_reranker: Reranker | None = None


def get_reranker(settings: Settings | None = None) -> Reranker:
    global _reranker
    settings = settings or get_settings()
    with _reranker_lock:
        if not settings.enable_reranking:
            return NoOpReranker()
        if _reranker is None:
            if settings.mock_models:
                _reranker = MockReranker(model_name=f"mock:{settings.reranker_model}")
            else:
                _reranker = CrossEncoderReranker(
                    settings.reranker_model, device=settings.llm_device
                )
        return _reranker


def reset_reranker() -> None:
    global _reranker
    with _reranker_lock:
        _reranker = None
