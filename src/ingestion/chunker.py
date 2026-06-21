"""
src/ingestion/chunker.py

Splits a PDFDocument into overlapping chunks ready for embedding.
Each chunk carries source + page metadata so citations work end-to-end.
"""
from __future__ import annotations
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
import config
from src.ingestion.pdf_loader import PDFDocument


@dataclass
class Chunk:
    """A single text chunk with full provenance metadata."""
    chunk_id:    str    # "{title}_p{page}_{idx}"
    text:        str
    source:      str    # original filename
    title:       str    # paper title (filename stem)
    page_number: int
    char_count:  int


class Chunker:
    """
    Splits PDFDocument pages into overlapping text chunks.

    Args:
        chunk_size:    Target token count per chunk (approx — splitter uses chars)
        chunk_overlap: Chars shared between consecutive chunks
    """

    # 1 token ≈ 4 chars — multiply token targets by 4 for char-based splitter
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        chunk_size:    int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size    = chunk_size    * self.CHARS_PER_TOKEN,
            chunk_overlap = chunk_overlap * self.CHARS_PER_TOKEN,
            # Try to split at paragraph → sentence → word boundaries
            separators    = ["\n\n", "\n", ". ", " ", ""],
            length_function = len,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_document(self, doc: PDFDocument) -> list[Chunk]:
        """Chunk a single PDFDocument. Returns a list of Chunk objects."""
        if doc.is_empty:
            logger.warning(f"Skipping empty document: {doc.title}")
            return []

        chunks = []
        global_idx = 0

        for page_number, page_text in doc.pages:
            page_chunks = self._splitter.split_text(page_text)

            for text in page_chunks:
                text = text.strip()
                if not text:
                    continue

                chunks.append(Chunk(
                    chunk_id    = f"{doc.title}_p{page_number}_{global_idx}",
                    text        = text,
                    source      = doc.metadata.get("source", doc.title),
                    title       = doc.title,
                    page_number = page_number,
                    char_count  = len(text),
                ))
                global_idx += 1

        logger.info(
            f"{doc.title}: {doc.page_count} pages → {len(chunks)} chunks "
            f"(avg {sum(c.char_count for c in chunks) // max(len(chunks),1)} chars/chunk)"
        )
        return chunks

    def chunk_documents(self, docs: list[PDFDocument]) -> list[Chunk]:
        """Chunk multiple documents. Returns flat list of all chunks."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(f"Total chunks across {len(docs)} documents: {len(all_chunks)}")
        return all_chunks


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from src.ingestion.pdf_loader import PDFLoader

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.chunker path/to/paper.pdf")
        sys.exit(1)

    loader  = PDFLoader()
    chunker = Chunker()

    doc    = loader.load_pdf(sys.argv[1])
    chunks = chunker.chunk_document(doc)

    print(f"\nTotal chunks : {len(chunks)}")
    print(f"\n--- Chunk 0 ---")
    print(f"ID     : {chunks[0].chunk_id}")
    print(f"Page   : {chunks[0].page_number}")
    print(f"Chars  : {chunks[0].char_count}")
    print(f"Text   :\n{chunks[0].text[:300]}")

    if len(chunks) > 1:
        print(f"\n--- Chunk 1 (check for overlap with Chunk 0) ---")
        print(chunks[1].text[:300])
