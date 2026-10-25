#!/usr/bin/env python3
"""Generate PDF/DOCX copies of sample TXT docs for format coverage demos."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "documents" / "sample"


def write_pdf(text: str, path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text[:4000], fontsize=10)
    remaining = text[4000:]
    while remaining:
        page = doc.new_page()
        page.insert_text((72, 72), remaining[:4000], fontsize=10)
        remaining = remaining[4000:]
    doc.save(path)
    doc.close()


def write_docx(text: str, path: Path) -> None:
    from docx import Document

    document = Document()
    for para in text.split("\n\n"):
        document.add_paragraph(para.strip())
    document.save(path)


def main() -> None:
    policy = (SAMPLE / "company_policy.txt").read_text(encoding="utf-8")
    support = (SAMPLE / "support_guide.txt").read_text(encoding="utf-8")
    write_pdf(policy, SAMPLE / "company_policy.pdf")
    write_docx(support, SAMPLE / "support_guide.docx")
    print("Wrote company_policy.pdf and support_guide.docx")


if __name__ == "__main__":
    main()
