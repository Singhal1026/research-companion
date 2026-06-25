"""
src/ingestion/pdf_loader.py

Loads PDFs and returns clean text with per-page metadata.
Uses PyMuPDF (fitz) — best choice for research papers.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF
from loguru import logger


@dataclass
class PDFDocument:
    """Represents a loaded PDF with extracted text and metadata."""

    file_name:    str
    file_path:    str
    pages:        list[tuple[int, str]] = field(default_factory=list)  # [(page_number, text), ...]
    total_pages:  int = 0
    failed_pages: list[int] = field(default_factory=list)              # pages that threw exceptions
    metadata:     dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """All page texts joined — used by chunker."""
        return "\n\n".join(text for _, text in self.pages)

    @property
    def is_empty(self) -> bool:
        """True if no text was extracted — likely a scanned PDF."""
        return len(self.pages) == 0

    @property
    def title(self) -> str:
        """Convenience accessor used by chunker and embedder."""
        return self.metadata.get("title") or Path(self.file_name).stem

    @property
    def page_count(self) -> int:
        """Number of pages that actually had extractable text."""
        return len(self.pages)


class PDFLoader:
    """
    Loads a single PDF or a folder of PDFs into PDFDocument objects.

    Args:
        min_page_chars: Pages with fewer chars after cleaning are skipped.
                        Catches blank pages, cover images, scanned pages.
        clean_text:     Set False to inspect raw extracted text for debugging.
    """

    def __init__(self, min_page_chars: int = 50, clean_text: bool = True):
        self.min_page_chars = min_page_chars
        self.clean_text     = clean_text

    # ── Public API ────────────────────────────────────────────────────────────

    def load_pdf(self, file_path: str | Path) -> Optional[PDFDocument]:
        """
        Load a single PDF. Returns None if the file is missing or unreadable
        so batch loading in load_folder() can skip bad files gracefully.
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        if path.suffix.lower() != ".pdf":
            logger.error(f"Not a PDF file: {file_path}")
            return None

        logger.info(f"Loading: {path.name}")

        doc = PDFDocument(file_name=path.name, file_path=str(path))

        try:
            with fitz.open(str(path)) as pdf:
                doc.total_pages = pdf.page_count
                doc.metadata    = self._extract_metadata(pdf, path)

                for page_num in range(pdf.page_count):
                    text = self._extract_page(pdf, page_num)

                    if text is None:
                        # Exception on this page — record it, keep going
                        doc.failed_pages.append(page_num + 1)
                        continue

                    if len(text) < self.min_page_chars:
                        logger.debug(
                            f"  Skipping page {page_num + 1} "
                            f"({len(text)} chars — blank or scanned)"
                        )
                        continue

                    if self.clean_text:
                        text = self._clean(text)

                    doc.pages.append((page_num + 1, text))  # 1-indexed

        except Exception as e:
            logger.error(f"Failed to open {path.name}: {e}")
            return None

        self._log_summary(doc)
        return doc

    def load_folder(self, folder_path: str | Path) -> list[PDFDocument]:
        """
        Load all PDFs in a folder (non-recursive). Skips files that fail.
        Catches both .pdf and .PDF extensions.
        """
        folder = Path(folder_path)

        if not folder.exists() or not folder.is_dir():
            logger.error(f"Folder not found: {folder_path}")
            return []

        # Catch both .pdf and .PDF
        pdf_files = sorted(set(folder.glob("*.pdf")) | set(folder.glob("*.PDF")))

        if not pdf_files:
            logger.warning(f"No PDF files found in: {folder_path}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF(s) in: {folder_path}")
        documents = []

        for pdf_path in pdf_files:
            doc = self.load_pdf(pdf_path)

            if doc and not doc.is_empty:
                documents.append(doc)
            elif doc and doc.is_empty:
                logger.warning(
                    f"  '{doc.file_name}' loaded but no text extracted — "
                    "likely a scanned PDF (needs OCR)"
                )

        logger.info(f"Successfully loaded {len(documents)}/{len(pdf_files)} PDFs")
        return documents

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_page(self, pdf: fitz.Document, page_num: int) -> Optional[str]:
        """Extract raw text from one page. Returns None if the page throws."""
        try:
            page = pdf.load_page(page_num)
            text = page.get_text("text")
            return text.strip() if text else None
        except Exception as e:
            logger.error(f"  Error on page {page_num + 1}: {e}")
            return None

    def _extract_metadata(self, pdf: fitz.Document, path: Path) -> dict:
        """
        Pull metadata from the PDF's own fields, with fallbacks.
        The 'source' and 'file_path' keys are used downstream
        by the embedder and retriever for citations.
        """
        raw = pdf.metadata or {}
        return {
            "title":      raw.get("title",    "").strip() or path.stem,
            "author":     raw.get("author",   "").strip() or "Unknown",
            "subject":    raw.get("subject",  "").strip() or "",
            "keywords":   raw.get("keywords", "").strip() or "",
            "page_count": pdf.page_count,
            # Keys the downstream pipeline relies on:
            "source":     path.name,
            "file_path":  str(path),
        }

    def _clean(self, text: str) -> str:
        """
        Clean raw PDF text. Handles the most common research paper issues:

        - Hyphenated line breaks: "sen-\\ntence" → "sentence"
          Uses lookbehind/lookahead so the word chars themselves aren't consumed.
        - Lone page numbers: lines that are just digits get stripped.
        - Trailing whitespace per line.
        - 3+ consecutive newlines collapsed to 2 (preserve paragraph spacing).
        """
        # Re-join hyphenated line breaks (lookbehind/lookahead — more precise)
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
        # Remove lines that are only digits 1–4 chars (page numbers)
        text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
        # Strip trailing whitespace on each line
        text = "\n".join(line.rstrip() for line in text.splitlines())
        # Collapse 3+ newlines → 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _log_summary(self, doc: PDFDocument) -> None:
        """One clean log line after each PDF is loaded."""
        status = "✓" if not doc.is_empty else "✗"
        logger.info(
            f"  {status} {doc.file_name} | "
            f"{doc.page_count}/{doc.total_pages} pages extracted | "
            f"~{len(doc.full_text):,} chars"
        )
        if doc.failed_pages:
            logger.warning(f"    Failed pages: {doc.failed_pages}")
        if doc.metadata.get("title") and doc.metadata["title"] != doc.file_name:
            logger.info(f"    Title: {doc.metadata['title']}")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.pdf_loader path/to/paper.pdf")
        sys.exit(1)

    loader = PDFLoader()
    doc    = loader.load_pdf(sys.argv[1])

    if doc is None:
        print("Failed to load PDF.")
        sys.exit(1)

    print(f"\nFile       : {doc.file_name}")
    print(f"Title      : {doc.title}")
    print(f"Author     : {doc.metadata.get('author')}")
    print(f"Pages      : {doc.page_count}/{doc.total_pages}")
    print(f"Total chars: {len(doc.full_text):,}")
    print(f"Is empty   : {doc.is_empty}")
    if doc.failed_pages:
        print(f"Failed pages: {doc.failed_pages}")
    print("\n--- First 500 chars ---")
    print(doc.full_text[:500])
