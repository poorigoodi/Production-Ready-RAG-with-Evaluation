"""SQLite metadata store for documents, chunks, conversations, and index versions."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import Settings, get_settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Thread-safe SQLite access with a single-writer friendly API."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    upload_timestamp TEXT NOT NULL,
                    page_count INTEGER,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    processing_status TEXT NOT NULL,
                    stored_path TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    page INTEGER,
                    text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS index_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    # --- Documents ---

    def get_document_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            return dict(row) if row else None

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY upload_timestamp DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def insert_document(self, doc: dict[str, Any]) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, filename, file_type, file_hash, upload_timestamp,
                    page_count, chunk_count, processing_status, stored_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc["document_id"],
                    doc["filename"],
                    doc["file_type"],
                    doc["file_hash"],
                    doc.get("upload_timestamp", _utc_now()),
                    doc.get("page_count"),
                    doc.get("chunk_count", 0),
                    doc["processing_status"],
                    doc["stored_path"],
                ),
            )

    def update_document_status(
        self,
        document_id: str,
        status: str,
        *,
        chunk_count: int | None = None,
        page_count: int | None = None,
    ) -> None:
        with self.connection() as conn:
            if chunk_count is not None and page_count is not None:
                conn.execute(
                    """
                    UPDATE documents
                    SET processing_status = ?, chunk_count = ?, page_count = ?
                    WHERE document_id = ?
                    """,
                    (status, chunk_count, page_count, document_id),
                )
            elif chunk_count is not None:
                conn.execute(
                    """
                    UPDATE documents
                    SET processing_status = ?, chunk_count = ?
                    WHERE document_id = ?
                    """,
                    (status, chunk_count, document_id),
                )
            else:
                conn.execute(
                    "UPDATE documents SET processing_status = ? WHERE document_id = ?",
                    (status, document_id),
                )

    def delete_document(self, document_id: str) -> bool:
        with self.connection() as conn:
            cur = conn.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
            return cur.rowcount > 0

    # --- Chunks ---

    def insert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT INTO chunks (chunk_id, document_id, page, text, chunk_index)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        c["chunk_id"],
                        c["document_id"],
                        c.get("page"),
                        c["text"],
                        c["chunk_index"],
                    )
                    for c in chunks
                ],
            )

    def delete_chunks_for_document(self, document_id: str) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    def list_chunks(
        self, document_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                rows = conn.execute(
                    f"""
                    SELECT c.*, d.filename
                    FROM chunks c
                    JOIN documents d ON d.document_id = c.document_id
                    WHERE c.document_id IN ({placeholders})
                    ORDER BY c.document_id, c.chunk_index
                    """,
                    document_ids,
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT c.*, d.filename
                    FROM chunks c
                    JOIN documents d ON d.document_id = c.document_id
                    ORDER BY c.document_id, c.chunk_index
                    """
                ).fetchall()
            return [dict(r) for r in rows]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        with self.connection() as conn:
            placeholders = ",".join("?" for _ in chunk_ids)
            rows = conn.execute(
                f"""
                SELECT c.*, d.filename
                FROM chunks c
                JOIN documents d ON d.document_id = c.document_id
                WHERE c.chunk_id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
            by_id = {r["chunk_id"]: dict(r) for r in rows}
            return [by_id[cid] for cid in chunk_ids if cid in by_id]

    # --- Conversations ---

    def create_conversation(self, conversation_id: str) -> None:
        now = _utc_now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (conversation_id, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                (conversation_id, now, now),
            )

    def conversation_exists(self, conversation_id: str) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return row is not None

    def delete_conversation(self, conversation_id: str) -> bool:
        with self.connection() as conn:
            cur = conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            return cur.rowcount > 0

    def add_message(
        self,
        message_id: str,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        now = _utc_now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (message_id, conversation_id, role, content, sources_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    json.dumps(sources or [], ensure_ascii=False),
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (now, conversation_id),
            )

    def get_recent_messages(
        self, conversation_id: str, limit: int
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
            messages = [dict(r) for r in reversed(rows)]
            for m in messages:
                m["sources"] = json.loads(m.pop("sources_json") or "[]")
            return messages

    # --- Index meta ---

    def set_index_meta(self, key: str, value: Any) -> None:
        payload = value if isinstance(value, str) else json.dumps(value, default=str)
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO index_meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, payload),
            )

    def get_index_meta(self, key: str) -> str | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT value FROM index_meta WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def get_index_meta_json(self, key: str) -> Any | None:
        raw = self.get_index_meta(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


_db: Database | None = None


def get_db(settings: Settings | None = None) -> Database:
    global _db
    if _db is None:
        settings = settings or get_settings()
        _db = Database(settings.sqlite_path)
    return _db


def reset_db_singleton() -> None:
    """Test helper to clear the process-wide database singleton."""
    global _db
    _db = None
