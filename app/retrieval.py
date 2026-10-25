"""FAISS dense retrieval, BM25 sparse retrieval, hybrid RRF fusion."""

from __future__ import annotations

import json
import logging
import pickle
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.chunking import Chunk
from app.config import Settings, get_settings
from app.database import Database, get_db
from app.embeddings import Embedder

logger = logging.getLogger(__name__)

RetrievalMode = Literal["dense", "sparse", "hybrid"]

# Persian / English tokenizer patterns
_TOKEN_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+|[A-Za-z0-9]+",
    re.UNICODE,
)


def normalize_for_bm25(text: str) -> str:
    """Normalize text for BM25 without destroying meaning."""
    text = text.replace("\ufeff", "")
    text = text.replace("\u200c", " ")  # ZWNJ -> space for token split
    text = text.replace("\u064a", "\u06cc")
    text = text.replace("\u0643", "\u06a9")
    text = text.lower()
    text = re.sub(r"[^\w\u0600-\u06FF\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Simple bilingual tokenizer (English + Persian).

    Limitations: no morphological stemming/lemmatization; compound Persian
    words may split awkwardly; English stemming is not applied.
    """
    normalized = normalize_for_bm25(text)
    return _TOKEN_RE.findall(normalized)


@dataclass
class ScoredChunk:
    chunk_id: str
    document_id: str
    document: str
    page: int | None
    text: str
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """RRF(d) = Σ 1 / (k + rank_i(d)) over ranked lists of chunk IDs."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class FaissIndexStore:
    """Persistent FAISS IndexFlatIP over L2-normalized vectors."""

    def __init__(self, index_dir: Path, dimension: int) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunk_ids: list[str] = []
        self.id_to_pos: dict[str, int] = {}

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def add(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        if len(chunk_ids) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dim mismatch: expected {self.dimension}, got {vectors.shape}"
            )
        # Ensure normalized for cosine via IP
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        vectors = vectors / norms
        self.index.add(vectors)
        for cid in chunk_ids:
            self.id_to_pos[cid] = len(self.chunk_ids)
            self.chunk_ids.append(cid)

    def search(
        self, query_vec: np.ndarray, top_k: int
    ) -> list[tuple[str, float]]:
        if self.size == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension {q.shape[1]} does not match index "
                f"dimension {self.dimension}. Rebuild the index after changing models."
            )
        norms = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        q = q / norms
        k = min(top_k, self.size)
        scores, indices = self.index.search(q, k)
        results: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append((self.chunk_ids[idx], float(score)))
        return results

    def save(self, basename: str = "faiss") -> None:
        """Atomic save: write to temp then replace."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir / f"{basename}.index"
        map_path = self.index_dir / f"{basename}_ids.json"
        with tempfile.TemporaryDirectory(dir=self.index_dir) as tmp:
            tmp_dir = Path(tmp)
            tmp_index = tmp_dir / "faiss.index"
            tmp_map = tmp_dir / "ids.json"
            faiss.write_index(self.index, str(tmp_index))
            tmp_map.write_text(
                json.dumps(
                    {"dimension": self.dimension, "chunk_ids": self.chunk_ids},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            shutil.move(str(tmp_index), str(index_path))
            shutil.move(str(tmp_map), str(map_path))

    @classmethod
    def load(cls, index_dir: Path, basename: str = "faiss") -> FaissIndexStore | None:
        index_dir = Path(index_dir)
        index_path = index_dir / f"{basename}.index"
        map_path = index_dir / f"{basename}_ids.json"
        if not index_path.exists() or not map_path.exists():
            return None
        try:
            meta = json.loads(map_path.read_text(encoding="utf-8"))
            store = cls(index_dir, int(meta["dimension"]))
            store.index = faiss.read_index(str(index_path))
            store.chunk_ids = list(meta["chunk_ids"])
            store.id_to_pos = {cid: i for i, cid in enumerate(store.chunk_ids)}
            if store.index.ntotal != len(store.chunk_ids):
                raise ValueError("FAISS ntotal does not match chunk id map")
            return store
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            raise


class BM25IndexStore:
    """In-memory BM25 over tokenized chunks; persisted via pickle for speed."""

    def __init__(self) -> None:
        self.chunk_ids: list[str] = []
        self.corpus_tokens: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        self.chunk_ids = list(chunk_ids)
        self.corpus_tokens = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self.bm25 or not self.chunk_ids:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        if len(scores) == 0:
            return []
        top_idx = np.argsort(scores)[::-1][:top_k]
        results: list[tuple[str, float]] = []
        for i in top_idx:
            score = float(scores[i])
            if score <= 0:
                continue
            results.append((self.chunk_ids[i], score))
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_ids": self.chunk_ids,
            "corpus_tokens": self.corpus_tokens,
        }
        with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False, suffix=".pkl"
        ) as tmp:
            pickle.dump(payload, tmp)
            tmp_path = Path(tmp.name)
        shutil.move(str(tmp_path), str(path))

    @classmethod
    def load(cls, path: Path) -> BM25IndexStore | None:
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            store = cls()
            store.chunk_ids = payload["chunk_ids"]
            store.corpus_tokens = payload["corpus_tokens"]
            store.bm25 = (
                BM25Okapi(store.corpus_tokens) if store.corpus_tokens else None
            )
            return store
        except Exception as exc:
            logger.error("Failed to load BM25 index: %s", exc)
            raise


class IndexManager:
    """Keeps FAISS + BM25 synchronized with SQLite chunk metadata."""

    def __init__(
        self,
        settings: Settings | None = None,
        db: Database | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.embedder = embedder
        self.faiss_store: FaissIndexStore | None = None
        self.bm25_store = BM25IndexStore()
        self._chunk_meta: dict[str, dict[str, Any]] = {}
        self._ready = False
        self._load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def attach_embedder(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def initialize(self) -> None:
        """Load existing indexes or rebuild from DB."""
        self._load_error = None
        try:
            meta = self.db.get_index_meta_json("embedding")
            if meta and self.embedder:
                if meta.get("model") != self.embedder.model_name:
                    logger.warning(
                        "Embedding model changed (%s -> %s). Rebuilding index.",
                        meta.get("model"),
                        self.embedder.model_name,
                    )
                    self.rebuild_from_db()
                    return
                if int(meta.get("dimension", -1)) != self.embedder.dimension:
                    logger.warning("Embedding dimension mismatch. Rebuilding index.")
                    self.rebuild_from_db()
                    return

            loaded = FaissIndexStore.load(self.settings.index_dir)
            bm25 = BM25IndexStore.load(self.settings.index_dir / "bm25.pkl")
            chunks = self.db.list_chunks()
            self._chunk_meta = {c["chunk_id"]: c for c in chunks}

            if loaded and bm25 and loaded.size == len(chunks):
                self.faiss_store = loaded
                self.bm25_store = bm25
                self._ready = True
                return

            # Rebuild if inconsistent or missing
            self.rebuild_from_db()
        except Exception as exc:
            self._ready = False
            self._load_error = str(exc)
            logger.error("Index initialization failed: %s", exc)
            raise

    def rebuild_from_db(self) -> None:
        if self.embedder is None:
            raise RuntimeError("Embedder required to rebuild index")
        chunks = self.db.list_chunks()
        self._chunk_meta = {c["chunk_id"]: c for c in chunks}
        dim = self.embedder.dimension
        self.faiss_store = FaissIndexStore(self.settings.index_dir, dim)
        if not chunks:
            self.bm25_store = BM25IndexStore()
            self._persist_meta()
            self._ready = True
            return

        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        vectors = self.embedder.embed_documents(texts)
        self.faiss_store.add(ids, vectors)
        self.bm25_store.build(ids, texts)
        self._persist_meta()
        self._ready = True

    def add_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if self.embedder is None:
            raise RuntimeError("Embedder not attached")
        if self.faiss_store is None:
            self.faiss_store = FaissIndexStore(
                self.settings.index_dir, self.embedder.dimension
            )
        if self.faiss_store.dimension != self.embedder.dimension:
            raise ValueError(
                "Cannot mix embedding dimensions. Rebuild the vector index."
            )
        ids = [c.chunk_id for c in chunks]
        self.faiss_store.add(ids, vectors)
        for c in chunks:
            self._chunk_meta[c.chunk_id] = {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "page": c.page,
                "text": c.text,
                "chunk_index": c.chunk_index,
                "filename": self.db.get_document(c.document_id)["filename"]
                if self.db.get_document(c.document_id)
                else "unknown",
            }
        # Rebuild BM25 deterministically from all chunks (simple & correct).
        all_chunks = self.db.list_chunks()
        self._chunk_meta = {c["chunk_id"]: c for c in all_chunks}
        self.bm25_store.build(
            [c["chunk_id"] for c in all_chunks],
            [c["text"] for c in all_chunks],
        )
        self._persist_meta()
        self._ready = True

    def _persist_meta(self) -> None:
        assert self.faiss_store is not None
        self.faiss_store.save()
        self.bm25_store.save(self.settings.index_dir / "bm25.pkl")
        self.db.set_index_meta(
            "embedding",
            {
                "model": self.embedder.model_name if self.embedder else None,
                "dimension": self.faiss_store.dimension,
                "normalized": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "index_version": "1",
                "chunk_count": self.faiss_store.size,
            },
        )

    def _hydrate(self, scored: list[tuple[str, float]], **score_fields: Any) -> list[ScoredChunk]:
        results: list[ScoredChunk] = []
        for chunk_id, score in scored:
            meta = self._chunk_meta.get(chunk_id)
            if not meta:
                continue
            results.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    document_id=meta["document_id"],
                    document=meta.get("filename", "unknown"),
                    page=meta.get("page"),
                    text=meta["text"],
                    score=score,
                    dense_score=score_fields.get("dense_scores", {}).get(chunk_id),
                    bm25_score=score_fields.get("bm25_scores", {}).get(chunk_id),
                )
            )
        return results

    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid",
        top_k: int = 5,
        rrf_k: int = 60,
        document_ids: list[str] | None = None,
        candidate_k: int | None = None,
    ) -> tuple[list[ScoredChunk], dict[str, float]]:
        if not self._ready or self.faiss_store is None:
            raise RuntimeError(
                self._load_error
                or "Search index is not ready. Rebuild from stored documents."
            )
        if self.embedder is None:
            raise RuntimeError("Embedder not available")

        latency: dict[str, float] = {
            "dense_ms": 0.0,
            "bm25_ms": 0.0,
            "fusion_ms": 0.0,
        }
        fetch_k = candidate_k or max(top_k * 4, top_k)

        dense_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        dense_ranked: list[str] = []
        sparse_ranked: list[str] = []

        if mode in ("dense", "hybrid"):
            t0 = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            dense_hits = self.faiss_store.search(qvec, fetch_k)
            latency["dense_ms"] = (time.perf_counter() - t0) * 1000
            dense_ranked = [cid for cid, _ in dense_hits]
            dense_scores = {cid: s for cid, s in dense_hits}

        if mode in ("sparse", "hybrid"):
            t0 = time.perf_counter()
            sparse_hits = self.bm25_store.search(query, fetch_k)
            latency["bm25_ms"] = (time.perf_counter() - t0) * 1000
            sparse_ranked = [cid for cid, _ in sparse_hits]
            bm25_scores = {cid: s for cid, s in sparse_hits}

        t0 = time.perf_counter()
        if mode == "dense":
            fused = [(cid, dense_scores[cid]) for cid in dense_ranked]
        elif mode == "sparse":
            fused = [(cid, bm25_scores[cid]) for cid in sparse_ranked]
        else:
            fused = reciprocal_rank_fusion(
                [dense_ranked, sparse_ranked], k=rrf_k
            )
        latency["fusion_ms"] = (time.perf_counter() - t0) * 1000

        allowed = set(document_ids) if document_ids else None
        filtered: list[tuple[str, float]] = []
        seen: set[str] = set()
        for cid, score in fused:
            if cid in seen:
                continue
            seen.add(cid)
            meta = self._chunk_meta.get(cid)
            if not meta:
                continue
            if allowed is not None and meta["document_id"] not in allowed:
                continue
            filtered.append((cid, score))
            if len(filtered) >= top_k:
                break

        results = self._hydrate(
            filtered,
            dense_scores=dense_scores,
            bm25_scores=bm25_scores,
        )
        return results, latency
