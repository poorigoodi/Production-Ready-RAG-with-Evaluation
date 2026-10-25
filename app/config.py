"""Application configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Validated runtime settings with sensible defaults."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Advanced RAG Evaluation"
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    streamlit_port: int = 8501
    cors_origins: str = "http://127.0.0.1:8501,http://localhost:8501"
    log_level: str = "INFO"

    embedding_model: str = "BAAI/bge-m3"
    embedding_batch_size: int = 32
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "Qwen/Qwen3-1.7B"
    llm_device: str = "auto"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.1
    mock_models: bool = False

    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)
    chunk_strategy: Literal["paragraph", "fixed"] = "paragraph"

    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, le=200)
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    enable_reranking: bool = True
    enable_query_rewriting: bool = False

    context_max_tokens: int = Field(default=2048, ge=256, le=32768)
    answer_reserve_tokens: int = Field(default=512, ge=64, le=8192)
    history_max_turns: int = Field(default=6, ge=0, le=50)
    min_retrieval_score: float = Field(default=0.05, ge=0.0, le=1.0)

    database_url: str = "sqlite:///./data/rag.db"
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    index_dir: Path = PROJECT_ROOT / "data" / "indexes"
    evaluation_dir: Path = PROJECT_ROOT / "data" / "evaluation"
    reports_dir: Path = PROJECT_ROOT / "reports"

    max_upload_size_mb: int = Field(default=20, ge=1, le=200)

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 800)
        if v >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return v

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def sqlite_path(self) -> Path:
        raw = self.database_url
        if raw.startswith("sqlite:///"):
            path = Path(raw.replace("sqlite:///", "", 1))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path
        return PROJECT_ROOT / "data" / "rag.db"

    def ensure_directories(self) -> None:
        for path in (
            self.documents_dir,
            self.index_dir,
            self.evaluation_dir,
            self.reports_dir / "metrics",
            self.reports_dir / "figures",
            self.sqlite_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
