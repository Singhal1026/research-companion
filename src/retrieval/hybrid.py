"""
src/retrieval/hybrid.py

Reciprocal Rank Fusion (RRF) of dense vector results and BM25 keyword results.

WHY HYBRID SEARCH:
  Dense search (vectors) captures semantic meaning — it finds chunks that
  mean the same thing even with different words. But it fails on exact
  terms: model names like "GPT-3", paper IDs like "1706.03762", author
  names, equation labels like "Equation 4", specific acronyms.

  BM25 (keyword) catches exact matches perfectly but misses semantic
  similarity. "attention mechanism" won't match "self-attention" via BM25
  unless those exact words appear.

  Hybrid search gets both. In production benchmarks (BEIR, MTEB) hybrid
  consistently outperforms either alone, especially on domain-specific
  corpora like research papers.

WHY RRF SPECIFICALLY:
  RRF (Cormack et al. 2009) combines ranked lists without needing to
  normalise scores across systems. Dense similarity (0–1) and BM25 scores
  (unbounded) can't be directly added — their scales are incompatible.
  RRF sidesteps this by using only rank position:

      RRF_score(chunk) = Σ  1 / (k + rank_in_list_i)

  where k=60 is the standard constant from the paper. A chunk ranked #1
  in both lists scores higher than a chunk ranked #1 in one and missing
  from the other. The formula is simple, parameter-free, and empirically
  robust.

FLOW:
  dense_results  (List[RetrievedChunk], sorted by dense_score desc)
  bm25_results   (List[RetrievedChunk], sorted by bm25_score desc)
       ↓
  RRF fusion  →  merged list scored by rrf_score
       ↓
  top-RRF_TOP_K results  →  reranker.py
"""
from __future__ import annotations
from typing import Optional

from rank_bm25 import BM25Okapi
from loguru import logger

import config
from src.retrieval.result import RetrievedChunk


class HybridFuser:
    """
    Fuses dense and BM25 retrieval results using Reciprocal Rank Fusion.

    Args:
        bm25:     The live BM25Okapi index. Get via embedder.get_bm25().
        all_docs: All documents stored in ChromaDB, in the same order as
                  the BM25 corpus. Get via collection.get()["documents"].
                  This parallel alignment is critical — BM25 returns indices,
                  and we use those indices to look up chunk metadata.
        top_k:    Number of fused results to return.
        rrf_k:    RRF constant. 60 is the standard from the original paper.
    """

    def __init__(
        self,
        bm25:     BM25Okapi,
        all_docs: list[str],          # parallel to BM25 corpus
        all_meta: list[dict],         # parallel metadata for each doc
        top_k: int = config.RRF_TOP_K,
        rrf_k: int = config.RRF_K,
    ):
        self._bm25     = bm25
        self._all_docs = all_docs
        self._all_meta = all_meta
        self._top_k    = top_k
        self._rrf_k    = rrf_k

    # ── Public API ────────────────────────────────────────────────────────────

    def fuse(
        self,
        query:         str,
        dense_results: list[RetrievedChunk],
        top_k:         Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Run BM25 search on query, then fuse with dense_results via RRF.

        Args:
            query:         The user query string.
            dense_results: Output of DenseRetriever.retrieve().
            top_k:         Override default top_k for this call.

        Returns:
            Merged list of RetrievedChunk sorted by rrf_score descending.
            Chunks appear once even if found by both methods.
            Both dense_score and bm25_score are populated on each chunk.
        """
        k = top_k or self._top_k

        if not self._all_docs:
            logger.warning("HybridFuser: BM25 corpus is empty — returning dense results only.")
            return dense_results[:k]

        # Step 1: BM25 search
        bm25_results = self._bm25_search(query, top_k=k)

        # Step 2: RRF fusion
        fused = self._rrf(dense_results, bm25_results)

        logger.debug(
            f"HybridFuser: dense={len(dense_results)}, "
            f"bm25={len(bm25_results)}, "
            f"fused={len(fused)} → top {k}"
        )
        return fused[:k]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """
        Run BM25 on the query and return top_k results as RetrievedChunk.

        BM25 returns a score array parallel to the corpus. We:
          1. Get scores for all documents
          2. Argsort to find top_k indices
          3. Look up text and metadata via the parallel all_docs/all_meta lists
          4. Build RetrievedChunk objects with bm25_score populated
        """
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # argsort ascending, take last top_k, reverse for descending
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                # BM25 score of 0 means no term overlap — not useful
                continue

            meta = self._all_meta[idx] if idx < len(self._all_meta) else {}
            text = self._all_docs[idx]  if idx < len(self._all_docs)  else ""

            results.append(RetrievedChunk(
                chunk_id         = meta.get("chunk_id", f"bm25_{idx}"),
                text             = text,
                title            = meta.get("title", ""),
                source           = meta.get("source", ""),
                page_number      = int(meta.get("page_number", 0)),
                bm25_score       = round(score, 4),
                retrieval_source = "bm25",
            ))

        return results

    def _rrf(
        self,
        dense_results: list[RetrievedChunk],
        bm25_results:  list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Reciprocal Rank Fusion.

        For each chunk_id, sum 1/(k + rank) across all lists it appears in.
        Chunks found by both methods score higher than those found by only one.

        The merged dict maps chunk_id → RetrievedChunk with:
          - dense_score from the dense list (if present)
          - bm25_score  from the BM25  list (if present)
          - rrf_score   = sum of 1/(k + rank) across all lists
          - retrieval_source = "dense", "bm25", or "both"
        """
        # chunk_id → RetrievedChunk (accumulates scores)
        merged: dict[str, RetrievedChunk] = {}

        # ── Score dense list ─────────────────────────────────────────────────
        for rank, chunk in enumerate(dense_results, start=1):
            rrf_contribution = 1.0 / (self._rrf_k + rank)
            if chunk.chunk_id not in merged:
                merged[chunk.chunk_id] = RetrievedChunk(
                    chunk_id    = chunk.chunk_id,
                    text        = chunk.text,
                    title       = chunk.title,
                    source      = chunk.source,
                    page_number = chunk.page_number,
                    dense_score = chunk.dense_score,
                    rrf_score   = rrf_contribution,
                    retrieval_source = "dense",
                )
            else:
                merged[chunk.chunk_id].dense_score = chunk.dense_score
                merged[chunk.chunk_id].rrf_score  += rrf_contribution

        # ── Score BM25 list ──────────────────────────────────────────────────
        for rank, chunk in enumerate(bm25_results, start=1):
            rrf_contribution = 1.0 / (self._rrf_k + rank)
            if chunk.chunk_id not in merged:
                merged[chunk.chunk_id] = RetrievedChunk(
                    chunk_id    = chunk.chunk_id,
                    text        = chunk.text,
                    title       = chunk.title,
                    source      = chunk.source,
                    page_number = chunk.page_number,
                    bm25_score  = chunk.bm25_score,
                    rrf_score   = rrf_contribution,
                    retrieval_source = "bm25",
                )
            else:
                # Chunk appeared in both lists — mark it and add BM25 score
                merged[chunk.chunk_id].bm25_score       = chunk.bm25_score
                merged[chunk.chunk_id].rrf_score       += rrf_contribution
                merged[chunk.chunk_id].retrieval_source = "both"

        # Sort by rrf_score descending
        fused = sorted(merged.values(), key=lambda c: c.rrf_score, reverse=True)

        # Round rrf_scores for readability
        for c in fused:
            c.rrf_score = round(c.rrf_score, 6)

        return fused