"""
src/retrieval/result.py

RetrievedChunk — the single result type that flows through the entire
retrieval pipeline: dense search → hybrid fusion → re-ranking → LLM.

WHY A SEPARATE FILE:
  retriever.py, hybrid.py, reranker.py, and pipeline.py all import this.
  Keeping it in its own file avoids circular imports and makes the type
  visible to the UI and agent layers without pulling in heavy dependencies.

SCORE FIELDS:
  Each stage adds its own score. By the time a chunk reaches the LLM,
  all four scores are populated. This is useful for:
    - Debugging why a chunk ranked highly (or didn't)
    - The UI: "This answer came from page 4 — confidence: 0.91"
    - RAGAS evaluation: correlating retrieval scores with answer quality

  dense_score:   Cosine similarity from ChromaDB (0.0–1.0, higher = more similar)
  bm25_score:    Raw BM25 score (unbounded, relative within a query)
  rrf_score:     Reciprocal Rank Fusion score (0.0–1.0 range after normalisation)
  rerank_score:  Cross-encoder score (unbounded logit, higher = more relevant)
                 Only populated after reranker.py runs.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """
    A single retrieved chunk with full provenance and per-stage scores.

    Flows through: retriever → hybrid → reranker → LLM context assembly.
    """
    # ── Content ───────────────────────────────────────────────────────────────
    chunk_id:    str
    text:        str
    title:       str    # paper title — shown in citations
    source:      str    # original filename — shown in citations
    page_number: int    # shown in citations: "page 4 of attention_paper.pdf"

    # ── Scores (populated stage by stage) ────────────────────────────────────
    dense_score:  float = 0.0   # set by retriever.py
    bm25_score:   float = 0.0   # set by hybrid.py
    rrf_score:    float = 0.0   # set by hybrid.py after RRF fusion
    rerank_score: float = 0.0   # set by reranker.py

    # ── Provenance ────────────────────────────────────────────────────────────
    # Which retrieval path surfaced this chunk. Useful for debugging.
    # Values: "dense", "bm25", "both"
    retrieval_source: str = "dense"

    def citation(self) -> str:
        """
        Human-readable citation string for the UI.
        Example: "attention_is_all_you_need.pdf — page 4"
        """
        return f"{self.source} — page {self.page_number}"

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk("
            f"title='{self.title}', "
            f"page={self.page_number}, "
            f"dense={self.dense_score:.3f}, "
            f"bm25={self.bm25_score:.3f}, "
            f"rrf={self.rrf_score:.3f}, "
            f"rerank={self.rerank_score:.3f})"
        )