"""Character-based chunking with fixed-size and paragraph-aware strategies.

Chunk size and overlap are measured in **characters**, not tokens.
Paragraph-aware chunking prefers splitting on blank lines / paragraph
boundaries while staying within the configured character budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ChunkStrategy = Literal["paragraph", "fixed"]


@dataclass
class TextBlock:
    """A contiguous text block with optional page reference."""

    text: str
    page: int | None = None


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    page: int | None
    text: str
    chunk_index: int


def make_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}_chunk_{chunk_index:03d}"


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Fixed-size character windows with overlap."""
    text = _normalize_whitespace(text)
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - chunk_overlap)
        if start >= end:
            start = end
    return chunks


def split_paragraph_aware(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Pack paragraphs into chunks up to chunk_size characters.

    Falls back to fixed splitting when a single paragraph exceeds chunk_size.
    Overlap is applied between consecutive chunks using trailing characters.
    """
    text = _normalize_whitespace(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return split_fixed(text, chunk_size, chunk_overlap)

    raw_chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            raw_chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for para in paragraphs:
        if len(para) > chunk_size:
            flush()
            raw_chunks.extend(split_fixed(para, chunk_size, chunk_overlap))
            continue
        extra = len(para) + (2 if current else 0)
        if current and current_len + extra > chunk_size:
            flush()
        current.append(para)
        current_len += extra if current_len else len(para)
    flush()

    if chunk_overlap <= 0 or len(raw_chunks) <= 1:
        return [c for c in raw_chunks if c]

    # Apply character overlap between chunks without re-duplicating whole chunks.
    overlapped: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
            continue
        prev_tail = raw_chunks[i - 1][-chunk_overlap:]
        merged = (prev_tail + "\n\n" + chunk).strip()
        if len(merged) > chunk_size + chunk_overlap:
            overlapped.append(chunk)
        else:
            overlapped.append(merged)
    return [c for c in overlapped if c]


def chunk_blocks(
    blocks: list[TextBlock],
    document_id: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    strategy: ChunkStrategy = "paragraph",
) -> list[Chunk]:
    """Chunk page/paragraph blocks while preserving page metadata when possible."""
    if not blocks:
        return []

    splitter = split_paragraph_aware if strategy == "paragraph" else split_fixed
    chunks: list[Chunk] = []
    index = 0

    # Group by page to preserve page references for PDF content.
    by_page: dict[int | None, list[str]] = {}
    order: list[int | None] = []
    for block in blocks:
        key = block.page
        if key not in by_page:
            by_page[key] = []
            order.append(key)
        if block.text.strip():
            by_page[key].append(block.text.strip())

    for page in order:
        page_text = "\n\n".join(by_page[page])
        pieces = splitter(page_text, chunk_size, chunk_overlap)
        for piece in pieces:
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document_id, index),
                    document_id=document_id,
                    page=page,
                    text=piece.strip(),
                    chunk_index=index,
                )
            )
            index += 1
    return chunks


def chunk_document_text(
    text: str,
    document_id: str,
    *,
    page: int | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    strategy: ChunkStrategy = "paragraph",
) -> list[Chunk]:
    return chunk_blocks(
        [TextBlock(text=text, page=page)],
        document_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy,
    )
