"""Document validation, extraction, and ingestion orchestration."""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

import pymupdf
from docx import Document as DocxDocument

from app.chunking import Chunk, TextBlock, chunk_blocks
from app.config import Settings, get_settings
from app.database import Database, get_db
from app.logging_utils import new_request_id, timed_operation

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
ABSTENTION_EMPTY = "Document contains no extractable text."


class IngestionError(Exception):
    """Raised when document ingestion fails validation or processing."""

    def __init__(self, message: str, category: str = "IngestionError") -> None:
        super().__init__(message)
        self.category = category


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_upload(
    filename: str,
    data: bytes,
    max_bytes: int,
) -> str:
    """Validate extension, size, and basic format. Returns normalized extension."""
    if not filename or "." not in filename:
        raise IngestionError("Missing file extension.", "ValidationError")

    # Prevent path traversal in stored names.
    safe_name = Path(filename).name
    if safe_name != filename.replace("\\", "/").split("/")[-1]:
        raise IngestionError("Invalid filename.", "ValidationError")

    ext = Path(safe_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported format '{ext}'. Allowed: PDF, DOCX, TXT.",
            "ValidationError",
        )

    if len(data) == 0:
        raise IngestionError("Uploaded file is empty.", "ValidationError")

    if len(data) > max_bytes:
        mb = max_bytes / (1024 * 1024)
        raise IngestionError(
            f"File exceeds maximum upload size of {mb:.0f} MB.",
            "ValidationError",
        )

    if ext == ".pdf" and not data.startswith(b"%PDF"):
        raise IngestionError("File extension is .pdf but content is not a PDF.", "ValidationError")
    if ext == ".docx" and data[:2] != b"PK":
        raise IngestionError(
            "File extension is .docx but content is not a ZIP/DOCX archive.",
            "ValidationError",
        )

    return ext


def normalize_text(text: str) -> str:
    """Light normalization for English and Persian without destructive changes."""
    text = text.replace("\ufeff", "")
    text = text.replace("\u200c", "\u200c")  # keep ZWNJ but normalize surroundings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize Arabic/Persian yeh/kaf variants carefully.
    text = text.replace("\u064a", "\u06cc")  # Arabic yeh -> Persian yeh
    text = text.replace("\u0643", "\u06a9")  # Arabic kaf -> Persian kaf
    return text.strip()


def extract_pdf(data: bytes) -> tuple[list[TextBlock], int]:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise IngestionError(f"Corrupted or unreadable PDF: {exc}", "ExtractionError") from exc

    blocks: list[TextBlock] = []
    try:
        for i, page in enumerate(doc, start=1):
            page_text = page.get_text("text") or ""
            page_text = normalize_text(page_text)
            if page_text:
                blocks.append(TextBlock(text=page_text, page=i))
        page_count = doc.page_count
    finally:
        doc.close()
    return blocks, page_count


def extract_docx(data: bytes) -> tuple[list[TextBlock], int | None]:
    from io import BytesIO

    try:
        document = DocxDocument(BytesIO(data))
    except Exception as exc:
        raise IngestionError(f"Corrupted or unreadable DOCX: {exc}", "ExtractionError") from exc

    paragraphs = [normalize_text(p.text) for p in document.paragraphs if p.text and p.text.strip()]
    if not paragraphs:
        # Tables as fallback
        for table in document.tables:
            for row in table.rows:
                cells = [normalize_text(c.text) for c in row.cells if c.text and c.text.strip()]
                if cells:
                    paragraphs.append(" | ".join(cells))
    text = "\n\n".join(paragraphs)
    if not text:
        return [], None
    return [TextBlock(text=text, page=None)], None


def extract_txt(data: bytes) -> tuple[list[TextBlock], int | None]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception as exc:
            raise IngestionError(f"Unable to decode text file: {exc}", "ExtractionError") from exc
    text = normalize_text(text)
    if not text:
        return [], None
    return [TextBlock(text=text, page=None)], None


def extract_text(ext: str, data: bytes) -> tuple[list[TextBlock], int | None]:
    if ext == ".pdf":
        return extract_pdf(data)
    if ext == ".docx":
        return extract_docx(data)
    if ext == ".txt":
        return extract_txt(data)
    raise IngestionError(f"Unsupported extension: {ext}", "ValidationError")


class DocumentIngestor:
    """Coordinates validation, extraction, chunking, embedding, and indexing."""

    def __init__(
        self,
        settings: Settings | None = None,
        db: Database | None = None,
        embedder: Any | None = None,
        index_manager: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db or get_db(self.settings)
        self.embedder = embedder
        self.index_manager = index_manager

    def set_services(self, embedder: Any, index_manager: Any) -> None:
        self.embedder = embedder
        self.index_manager = index_manager

    def ingest_bytes(
        self,
        filename: str,
        data: bytes,
        *,
        request_id: str | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or new_request_id()
        with timed_operation(
            logger, request_id=request_id, operation="ingest_document", filename=Path(filename).name
        ):
            return self._ingest(
                filename, data, request_id=request_id, document_id=document_id
            )

    def _ingest(
        self,
        filename: str,
        data: bytes,
        *,
        request_id: str,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        safe_name = Path(filename).name
        ext = validate_upload(safe_name, data, self.settings.max_upload_bytes)
        file_hash = sha256_bytes(data)

        existing = self.db.get_document_by_hash(file_hash)
        if existing:
            return {
                "document_id": existing["document_id"],
                "filename": existing["filename"],
                "chunk_count": existing["chunk_count"],
                "page_count": existing["page_count"],
                "status": existing["processing_status"],
                "duplicate": True,
                "message": "Document already indexed (duplicate content hash).",
            }

        if self.embedder is None or self.index_manager is None:
            raise IngestionError(
                "Embedding/index services are not initialized.",
                "ServiceUnavailable",
            )

        document_id = document_id or f"doc_{uuid.uuid4().hex[:12]}"
        if self.db.get_document(document_id):
            raise IngestionError(
                f"Document ID already exists: {document_id}",
                "ValidationError",
            )
        stored_path = self.settings.documents_dir / f"{document_id}{ext}"

        self.db.insert_document(
            {
                "document_id": document_id,
                "filename": safe_name,
                "file_type": ext.lstrip("."),
                "file_hash": file_hash,
                "page_count": None,
                "chunk_count": 0,
                "processing_status": "processing",
                "stored_path": str(stored_path),
            }
        )

        try:
            stored_path.write_bytes(data)
            blocks, page_count = extract_text(ext, data)
            if not blocks or not any(b.text.strip() for b in blocks):
                raise IngestionError(ABSTENTION_EMPTY, "EmptyDocument")

            chunks = chunk_blocks(
                blocks,
                document_id,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                strategy=self.settings.chunk_strategy,
            )
            if not chunks:
                raise IngestionError(ABSTENTION_EMPTY, "EmptyDocument")

            texts = [c.text for c in chunks]
            t0 = time.perf_counter()
            vectors = self.embedder.embed_documents(texts)
            embed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                '{"request_id":"%s","operation":"embed_documents","duration_ms":%.2f,"chunks":%d}',
                request_id,
                embed_ms,
                len(chunks),
            )

            self.db.insert_chunks(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "page": c.page,
                        "text": c.text,
                        "chunk_index": c.chunk_index,
                    }
                    for c in chunks
                ]
            )
            self.index_manager.add_chunks(chunks, vectors)
            self.db.update_document_status(
                document_id,
                "indexed",
                chunk_count=len(chunks),
                page_count=page_count,
            )
            return {
                "document_id": document_id,
                "filename": safe_name,
                "chunk_count": len(chunks),
                "page_count": page_count,
                "status": "indexed",
                "duplicate": False,
                "message": None,
            }
        except Exception:
            self._rollback_document(document_id, stored_path)
            raise

    def _rollback_document(self, document_id: str, stored_path: Path) -> None:
        try:
            self.db.delete_chunks_for_document(document_id)
            self.db.delete_document(document_id)
        except Exception:
            logger.exception("Failed to rollback document metadata for %s", document_id)
        try:
            if stored_path.exists():
                stored_path.unlink()
        except Exception:
            logger.exception("Failed to remove stored file for %s", document_id)
        try:
            if self.index_manager is not None:
                self.index_manager.rebuild_from_db()
        except Exception:
            logger.exception("Failed to rebuild indexes after rollback for %s", document_id)

    def delete_document(self, document_id: str) -> bool:
        doc = self.db.get_document(document_id)
        if not doc:
            return False
        stored = Path(doc["stored_path"])
        self.db.delete_chunks_for_document(document_id)
        self.db.delete_document(document_id)
        if stored.exists():
            try:
                stored.unlink()
            except OSError:
                logger.warning("Could not delete stored file %s", stored)
        if self.index_manager is not None:
            self.index_manager.rebuild_from_db()
        return True


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "page": c.page,
            "text": c.text,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
