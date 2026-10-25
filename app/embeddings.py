"""Sentence-Transformers embedding generation with mock support for tests."""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Protocol

import numpy as np

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def detect_device(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


class MockEmbedder:
    """Deterministic hash-based embeddings for tests (no model download)."""

    def __init__(self, model_name: str = "mock-embedder", dimension: int = 384) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.normalized = True

    def _embed_one(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        rng_seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(rng_seed)
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        # Mix in token presence for crude lexical signal in tests.
        for token in text.lower().split():
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension
            vec[idx] += 1.0
        norm = np.linalg.norm(vec) + 1e-12
        return (vec / norm).astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)


class SentenceTransformerEmbedder:
    """Production embedder backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = detect_device(device)
        self.batch_size = batch_size
        self.normalized = normalize
        logger.info("Loading embedding model %s on %s", model_name, self.device)
        self._model = SentenceTransformer(model_name, device=self.device)
        # Probe dimension
        probe = self._model.encode(
            ["dimension probe"],
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        self.dimension = int(probe.shape[1])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalized,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vectors = self._model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=self.normalized,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors[0], dtype=np.float32)


_embedder_lock = threading.Lock()
_embedder: Embedder | None = None


def get_embedder(settings: Settings | None = None) -> Embedder:
    global _embedder
    settings = settings or get_settings()
    with _embedder_lock:
        if _embedder is not None:
            if _embedder.model_name != settings.embedding_model and not settings.mock_models:
                # Model change requires explicit rebuild; recreate embedder.
                _embedder = None
        if _embedder is None:
            if settings.mock_models:
                _embedder = MockEmbedder(model_name=f"mock:{settings.embedding_model}")
            else:
                _embedder = SentenceTransformerEmbedder(
                    settings.embedding_model,
                    device=settings.llm_device,
                    batch_size=settings.embedding_batch_size,
                    normalize=True,
                )
        return _embedder


def reset_embedder() -> None:
    global _embedder
    with _embedder_lock:
        _embedder = None
