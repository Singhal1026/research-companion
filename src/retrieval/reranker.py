"""
src/retrieval/reranker.py

Cross-encoder re-ranking of the top-N hybrid search results.

WHY RE-RANKING:
  Bi-encoder retrieval (what sentence-transformers does) encodes query and
  chunk INDEPENDENTLY and compares their embeddings. This is fast — O(1)
  at query time after indexing — but loses the interaction between query
  and chunk words.

  A cross-encoder processes the query and chunk TOGETHER as a single
  input: [CLS] query [SEP] chunk [SEP]. It can see which words in the
  chunk actually answer the query. This is much more accurate but O(n)
  at query time, so it only runs on the top-N candidates from hybrid search
  (typically 20), not the full index.

  Typical pipeline:
    Full index  →  bi-encoder retrieval (fast, approximate)  →  top-20
    top-20      →  cross-encoder re-ranking (slow, accurate)  →  top-4
    top-4       →  LLM context window

  The re-ranker is the quality gate. Everything before it is about recall
  (don't miss relevant chunks). Re-ranking is about precision (only give
  the LLM the best chunks).

MODEL CHOICE:
  cross-encoder/ms-marco-MiniLM-L-6-v2
    - Trained on MS MARCO passage ranking dataset
    - 22M parameters — fast on CPU (~50ms for 20 pairs)
    - Returns logits (unbounded floats, higher = more relevant)
    - Good enough for research paper QA

  Alternative for higher quality: cross-encoder/ms-marco-MiniLM-L-12-v2
  (same architecture, twice as many layers, slower but more accurate)
"""
from __future__ import annotations
from typing import Optional

from sentence_transformers.cross_encoder import CrossEncoder
from loguru import logger

import config
from src.retrieval.result import RetrievedChunk


class Reranker:
    """
    Re-ranks a list of RetrievedChunks using a cross-encoder model.

    Args:
        model_name: HuggingFace cross-encoder model name.
        top_k:      Number of chunks to return after re-ranking.
    """

    def __init__(
        self,
        model_name: str = config.RERANKER_MODEL,
        top_k:      int = config.RERANK_TOP_K,
    ):
        self._top_k = top_k
        logger.info(f"Loading re-ranker: {model_name}")
        self._model = CrossEncoder(model_name)
        logger.info("Re-ranker ready")

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query:   str,
        chunks:  list[RetrievedChunk],
        top_k:   Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Re-score chunks using a cross-encoder and return the top-k.

        Args:
            query:  The user query string.
            chunks: Output of HybridFuser.fuse() — the candidates to re-rank.
            top_k:  Override default top_k.

        Returns:
            Top-k RetrievedChunk objects sorted by rerank_score descending.
            rerank_score is populated on each returned chunk.
            All other score fields (dense, bm25, rrf) are preserved.
        """
        k = top_k or self._top_k

        if not chunks:
            return []

        # Cross-encoder expects (query, passage) pairs
        pairs = [(query, chunk.text) for chunk in chunks]

        # predict() returns a numpy array of logit scores, one per pair
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Attach rerank_score to each chunk (preserve all other scores)
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = round(float(score), 4)

        # Sort by rerank_score descending, return top-k
        reranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)

        logger.debug(
            f"Reranker: {len(chunks)} candidates → top {k} | "
            f"scores [{reranked[0].rerank_score:.3f} … "
            f"{reranked[min(k-1, len(reranked)-1)].rerank_score:.3f}]"
        )
        return reranked[:k]