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
    """Structured output from the PDF loader."""
    file_path:  str
    title:      str
    page_count: int
    pages:      list[tuple[int, str]]   # [(page_number, text), ...]
    full_text:  str
    is_empty:   bool = False            # True if scanned / no text layer
    metadata:   dict = field(default_factory=dict)


class PDFLoader:
    """
    Loads a single PDF or a folder of PDFs into PDFDocument objects.

    Args:
        min_page_chars: Pages with fewer chars after cleaning are skipped.
                        Catches cover pages, figure-only pages, page numbers.
    """

    def __init__(self, min_page_chars: int = 50):
        self.min_page_chars = min_page_chars

    # ── Public API ────────────────────────────────────────────────────────────

    def load_pdf(self, path: str | Path) -> PDFDocument:
        """Load a single PDF. Returns a PDFDocument."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        logger.info(f"Loading: {path.name}")

        doc = fitz.open(str(path))
        pages = self._extract_pages(doc)
        doc.close()

        full_text = "\n\n".join(text for _, text in pages)
        is_empty  = len(pages) == 0

        if is_empty:
            logger.warning(
                f"{path.name} appears to be a scanned PDF (no text layer). "
                "OCR would be needed to extract text."
            )

        return PDFDocument(
            file_path  = str(path),
            title      = path.stem,
            page_count = len(pages),
            pages      = pages,
            full_text  = full_text,
            is_empty   = is_empty,
            metadata   = {"source": path.name, "file_path": str(path)},
        )

    def load_folder(self, folder: str | Path) -> list[PDFDocument]:
        """Load all PDFs in a folder. Skips files that fail."""
        folder = Path(folder)
        pdf_files = sorted(folder.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {folder}")
            return []

        logger.info(f"Found {len(pdf_files)} PDFs in {folder}")
        documents = []

        for pdf_path in pdf_files:
            try:
                doc = self.load_pdf(pdf_path)
                if not doc.is_empty:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load {pdf_path.name}: {e}")

        logger.info(f"Successfully loaded {len(documents)}/{len(pdf_files)} PDFs")
        return documents

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_pages(self, doc: fitz.Document) -> list[tuple[int, str]]:
        """Extract and clean text from each page. Returns 1-indexed page list."""
        pages = []
        for i, page in enumerate(doc):
            raw  = page.get_text("text")
            text = self._clean(raw)
            if len(text) >= self.min_page_chars:
                pages.append((i + 1, text))  # 1-indexed page number
        return pages

    def _clean(self, text: str) -> str:
        """
        Clean raw PDF text. Handles the most common research paper extraction issues:
        - Hyphenated line breaks:  "sen-\ntence"  →  "sentence"
        - Lone page numbers:       lines that are just a digit get stripped
        - Excess whitespace
        """
        # Re-join hyphenated line breaks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # Remove lines that are only digits (page numbers)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.splitlines()]
        text  = "\n".join(lines)
        return text.strip()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.pdf_loader path/to/paper.pdf")
        sys.exit(1)

    loader = PDFLoader()
    doc    = loader.load_pdf(sys.argv[1])

    print(f"\nTitle      : {doc.title}")
    print(f"Pages      : {doc.page_count}")
    print(f"Total chars: {len(doc.full_text):,}")
    print(f"Is empty   : {doc.is_empty}")
    print("\n--- First 500 chars ---")
    print(doc.full_text[:500])
