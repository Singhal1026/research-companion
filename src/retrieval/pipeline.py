"""
src/retrieval/pipeline.py

RetrievalPipeline — the single entry point for all retrieval.

The agent, UI, and any other caller only ever calls:

    results = pipeline.retrieve(query, route_result)

Internally it routes to the right strategy:

    DIRECT      →  [] (no retrieval — LLM answers from training data)
    SINGLE_PAPER →  dense retrieval filtered to one document → reranker
    HYBRID      →  dense + BM25 → RRF fusion → reranker    (main path)
    AGENT       →  same as HYBRID (agent does multi-hop calls itself)

WHY THIS FILE EXISTS:
  Without a pipeline wrapper, every caller (Gradio UI, LangGraph agent,
  CLI test script) would have to manually orchestrate retriever → hybrid
  → reranker and handle the route logic themselves. That's three separate
  concerns duplicated across callers. The pipeline hides the complexity
  behind one method.

  It also makes the retrieval strategy swappable — to add a new route or
  change the fusion algorithm, you change it in one place.
"""
from __future__ import annotations

from loguru import logger
from sentence_transformers import SentenceTransformer

import config
from src.retrieval.result     import RetrievedChunk
from src.retrieval.router     import RouterResult, Route
from src.retrieval.retriever  import DenseRetriever
from src.retrieval.hybrid     import HybridFuser
from src.retrieval.reranker   import Reranker


class RetrievalPipeline:
    """
    Orchestrates the full retrieval stack: dense → hybrid → rerank.

    Constructed once at app startup and reused for every query.

    Args:
        embedder: The live Embedder instance. Pipeline calls
                  embedder.get_collection() and embedder.get_bm25()
                  to stay in sync after new documents are indexed.

    Usage:
        pipeline = RetrievalPipeline(embedder)
        results  = pipeline.retrieve("what is attention?", router_result)
        # results: List[RetrievedChunk], ready for LLM context assembly
    """

    def __init__(self, embedder):   # type - Embedder (avoid circular import)
        self._embedder = embedder

        # Load the embedding model once — shared with Embedder
        logger.info("RetrievalPipeline: loading embedding model…")
        self._model = SentenceTransformer(config.EMBEDDING_MODEL)

        # Reranker loads the cross-encoder model at init
        self._reranker = Reranker()

        logger.info("RetrievalPipeline ready")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:        str,
        route_result: RouterResult,
    ) -> list[RetrievedChunk]:
        """
        Run the retrieval strategy appropriate for the given route.

        Args:
            query:        The user query string.
            route_result: Output of QueryRouter.classify(query).

        Returns:
            List of RetrievedChunk ready for LLM context assembly.
            Empty list for DIRECT route (LLM uses its own knowledge).
        """
        route = route_result.route

        if route == Route.DIRECT:
            logger.debug("RetrievalPipeline: DIRECT route — skipping retrieval")
            return []

        elif route == Route.SINGLE_PAPER:
            return self._retrieve_single_paper(query, route_result.target_paper)

        elif route in (Route.HYBRID, Route.AGENT):
            # AGENT uses the same retrieval as HYBRID.
            # The agent calls retrieve() multiple times with targeted sub-queries.
            return self._retrieve_hybrid(query)

        else:
            logger.warning(f"RetrievalPipeline: unknown route '{route}' — defaulting to hybrid")
            return self._retrieve_hybrid(query)

    # ── Retrieval strategies ──────────────────────────────────────────────────

    def _retrieve_hybrid(self, query: str) -> list[RetrievedChunk]:
        """
        Full pipeline: dense → BM25 → RRF fusion → cross-encoder rerank.
        This is the main production path for most queries.
        """
        collection = self._embedder.get_collection()
        bm25       = self._embedder.get_bm25()

        # Step 1: Dense retrieval
        retriever     = DenseRetriever(self._model, collection)
        dense_results = retriever.retrieve(query)

        # Step 2: Hybrid fusion (needs BM25 + all corpus docs + metadata)
        if bm25 is None:
            logger.warning(
                "RetrievalPipeline: BM25 index not ready — "
                "returning dense-only results (reranked)"
            )
            return self._reranker.rerank(query, dense_results)

        # Fetch all docs + metadata for BM25 lookup
        # (parallel to BM25 corpus built during indexing)
        all_data = collection.get(include=["documents", "metadatas"])
        all_docs = all_data.get("documents") or []
        all_meta = all_data.get("metadatas") or []

        fuser        = HybridFuser(bm25, all_docs, all_meta)
        fused_results = fuser.fuse(query, dense_results)

        # Step 3: Cross-encoder re-ranking
        return self._reranker.rerank(query, fused_results)

    def _retrieve_single_paper(
        self,
        query:        str,
        target_paper: str | None,
    ) -> list[RetrievedChunk]:
        """
        Dense retrieval scoped to one document, then reranked.
        Used when the router identifies a specific paper in the query.
        """
        collection = self._embedder.get_collection()
        retriever  = DenseRetriever(self._model, collection)

        if not target_paper:
            # Router said SINGLE_PAPER but didn't identify which one.
            # Fall back to full hybrid — better than an empty result.
            logger.warning(
                "RetrievalPipeline: SINGLE_PAPER route but no target_paper set "
                "— falling back to hybrid"
            )
            return self._retrieve_hybrid(query)

        dense_results = retriever.retrieve(query, filter_source=target_paper)

        if not dense_results:
            logger.warning(
                f"RetrievalPipeline: no chunks found for '{target_paper}' "
                "— is it indexed?"
            )
            return []

        return self._reranker.rerank(query, dense_results)