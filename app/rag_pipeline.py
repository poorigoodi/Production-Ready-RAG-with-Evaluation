"""End-to-end RAG orchestration: retrieve → rerank → generate → cite."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from app.config import Settings, get_settings
from app.database import Database, get_db
from app.embeddings import Embedder, get_embedder
from app.generation import GroundedGenerator, get_llm
from app.logging_utils import new_request_id, timed_operation
from app.reranking import Reranker, get_reranker
from app.retrieval import IndexManager, RetrievalMode, ScoredChunk

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        db: Database | None = None,
        embedder: Embedder | None = None,
        index_manager: IndexManager | None = None,
        reranker: Reranker | None = None,
        generator: GroundedGenerator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.embedder = embedder
        self.index_manager = index_manager
        self.reranker = reranker
        self.generator = generator

    def bind_services(
        self,
        embedder: Embedder,
        index_manager: IndexManager,
        reranker: Reranker,
        generator: GroundedGenerator,
    ) -> None:
        self.embedder = embedder
        self.index_manager = index_manager
        self.reranker = reranker
        self.generator = generator

    def rewrite_query(self, question: str, history: list[dict]) -> str:
        """Lightweight history-aware rewrite without inventing facts."""
        if not self.settings.enable_query_rewriting or not history:
            return question
        # Use last user+assistant turns as context anchors.
        prior_user = [
            m["content"] for m in history if m.get("role") == "user"
        ]
        if not prior_user:
            return question
        # Pronoun / ellipsis style follow-ups
        followup_markers = (
            "it",
            "that",
            "this",
            "they",
            "those",
            "does it",
            "is it",
            "what about",
            "and the",
            "آن",
            "این",
            "آیا",
        )
        q_low = question.lower()
        if not any(m in q_low for m in followup_markers) and len(question.split()) > 6:
            return question
        topic = prior_user[-1]
        return f"{question} (regarding: {topic})"

    def search(
        self,
        query: str,
        *,
        retrieval_mode: RetrievalMode | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
        document_ids: list[str] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or new_request_id()
        mode = retrieval_mode or self.settings.retrieval_mode
        k = top_k or self.settings.top_k
        do_rerank = self.settings.enable_reranking if rerank is None else rerank

        with timed_operation(
            logger, request_id=request_id, operation="search", mode=mode
        ):
            assert self.index_manager is not None
            candidate_k = (
                max(k, self.settings.rerank_candidates) if do_rerank else k
            )
            results, latency = self.index_manager.retrieve(
                query,
                mode=mode,
                top_k=candidate_k,
                rrf_k=self.settings.rrf_k,
                document_ids=document_ids,
                candidate_k=candidate_k,
            )
            rerank_ms = 0.0
            if do_rerank and self.reranker is not None and self.reranker.enabled:
                results, rerank_ms = self.reranker.rerank(query, results, k)
            else:
                results = results[:k]
            total = sum(latency.values()) + rerank_ms
            return {
                "query": query,
                "results": results,
                "latency_ms": total,
                "latency": {**latency, "rerank_ms": rerank_ms},
                "retrieval_mode": mode,
                "reranked": bool(do_rerank and self.reranker and self.reranker.enabled),
            }

    def chat(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        retrieval_mode: RetrievalMode | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or new_request_id()
        t_all = time.perf_counter()

        if not conversation_id:
            conversation_id = f"conversation-{uuid.uuid4().hex[:12]}"
            self.db.create_conversation(conversation_id)
        elif not self.db.conversation_exists(conversation_id):
            self.db.create_conversation(conversation_id)

        history = self.db.get_recent_messages(
            conversation_id, limit=self.settings.history_max_turns * 2
        )
        rewritten = self.rewrite_query(question, history)

        search_out = self.search(
            rewritten,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            rerank=rerank,
            request_id=request_id,
        )
        chunks: list[ScoredChunk] = search_out["results"]

        assert self.generator is not None
        answer, selected, abstained, gen_ms = self.generator.generate_answer(
            question,
            chunks,
            history_messages=history,
        )

        sources = [
            {
                "document": c.document,
                "page": c.page,
                "chunk_id": c.chunk_id,
                "score": c.score,
                "text_preview": c.text[:200] + ("…" if len(c.text) > 200 else ""),
            }
            for c in selected
        ]

        self.db.add_message(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            role="user",
            content=question,
        )
        self.db.add_message(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            sources=sources,
        )

        total_ms = (time.perf_counter() - t_all) * 1000
        latency = search_out["latency"]
        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": total_ms,
            "conversation_id": conversation_id,
            "abstained": abstained,
            "rewritten_query": rewritten if rewritten != question else None,
            "latency_breakdown": {
                "dense_ms": latency.get("dense_ms"),
                "bm25_ms": latency.get("bm25_ms"),
                "fusion_ms": latency.get("fusion_ms"),
                "rerank_ms": latency.get("rerank_ms"),
                "generation_ms": gen_ms,
                "total_ms": total_ms,
            },
        }

    def clear_conversation(self, conversation_id: str) -> bool:
        return self.db.delete_conversation(conversation_id)


def create_default_pipeline(settings: Settings | None = None) -> RAGPipeline:
    settings = settings or get_settings()
    db = get_db(settings)
    embedder = get_embedder(settings)
    index_manager = IndexManager(settings, db, embedder)
    index_manager.initialize()
    reranker = get_reranker(settings)
    llm = get_llm(settings)
    generator = GroundedGenerator(llm, settings)
    pipeline = RAGPipeline(settings, db)
    pipeline.bind_services(embedder, index_manager, reranker, generator)
    return pipeline
