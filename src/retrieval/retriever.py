"""
src/retrieval/retriever.py

Dense vector retrieval against ChromaDB.
Embeds the query, queries the collection, returns RetrievedChunk objects.

DESIGN DECISIONS:

  Query embedding is normalised (same as index-time)
    At index time, embedder.py uses normalize_embeddings=True. The query
    must be normalised too. Mismatched normalisation silently degrades
    recall — cosine similarity assumes both vectors are unit-length.

  filter_source for SINGLE_PAPER route
    When the router returns Route.SINGLE_PAPER, the retriever scopes
    the search to one document via ChromaDB's where= metadata filter.
    This is faster and more precise than cross-paper search for targeted
    questions like "what methodology did the BERT paper use?".

  include_distances=True
    ChromaDB returns distances, not similarities. For cosine space:
        similarity = 1 - distance
    We convert at retrieval time so all downstream stages work with
    similarity scores (higher = better), which is the intuitive direction.

  Deduplication
    ChromaDB can return the same chunk_id twice in rare cases (index
    corruption, concurrent writes). We deduplicate by chunk_id before
    returning. Silent duplicates in the context window confuse the LLM.

  Empty collection guard
    If the collection has zero chunks (nothing indexed yet), return []
    immediately with a clear warning rather than letting ChromaDB throw
    a confusing internal error.
"""
from __future__ import annotations
from typing import Optional

from sentence_transformers import SentenceTransformer
from loguru import logger

import config
from src.retrieval.result import RetrievedChunk


class DenseRetriever:
    """
    Queries ChromaDB with a dense vector embedding of the query.

    Args:
        model:      sentence-transformers model. Pass the same instance
                    used by the Embedder to avoid loading it twice.
        collection: ChromaDB collection. Get via embedder.get_collection().
        top_k:      Number of results to return per query.
    """

    def __init__(
        self,
        model:      SentenceTransformer,
        collection,                         # chromadb.Collection
        top_k: int = config.RETRIEVAL_TOP_K,
    ):
        self._model      = model
        self._collection = collection
        self._top_k      = top_k

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:         str,
        top_k:         Optional[int] = None,
        filter_source: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Embed the query and retrieve the top-k most similar chunks.

        Args:
            query:         User query string.
            top_k:         Override the default top_k for this call.
            filter_source: If set, only return chunks where
                           metadata['title'] == filter_source.
                           Used by the SINGLE_PAPER route.

        Returns:
            List of RetrievedChunk sorted by dense_score descending.
            Empty list if the collection is empty or query fails.
        """
        k = top_k or self._top_k

        # Guard: empty collection produces a confusing ChromaDB error
        if self._collection.count() == 0:
            logger.warning("DenseRetriever: collection is empty — nothing to retrieve.")
            return []

        # Clamp k to collection size (ChromaDB errors if k > n_docs)
        k = min(k, self._collection.count())

        query = query.strip()
        if not query:
            logger.warning("DenseRetriever: empty query string.")
            return []

        # Embed query — must match index-time normalisation
        query_embedding = self._model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # Build ChromaDB query kwargs
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results":        k,
            "include":          ["documents", "metadatas", "distances"],
        }

        # SINGLE_PAPER filter — scope search to one document
        if filter_source:
            query_kwargs["where"] = {"title": filter_source}
            logger.debug(f"DenseRetriever: scoped to source='{filter_source}'")

        try:
            results = self._collection.query(**query_kwargs)
        except Exception as e:
            logger.error(f"DenseRetriever: ChromaDB query failed: {e}")
            return []

        chunks = self._parse_results(results)
        chunks = self._deduplicate(chunks)

        logger.debug(
            f"DenseRetriever: '{query[:60]}' → {len(chunks)} results"
            + (f" (filtered to '{filter_source}')" if filter_source else "")
        )
        return chunks

    # ── Internal ──────────────────────────────────────────────────────────────

    def _parse_results(self, results: dict) -> list[RetrievedChunk]:
        """
        Convert ChromaDB query results into RetrievedChunk objects.

        ChromaDB returns nested lists (one list per query — we only send one).
        Distances are converted to similarities: similarity = 1 - distance.
        """
        ids       = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        chunks = []
        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            # cosine distance → cosine similarity
            # ChromaDB cosine distance is in [0, 2]; typical range is [0, 1]
            similarity = max(0.0, 1.0 - dist)

            chunks.append(RetrievedChunk(
                chunk_id         = chunk_id,
                text             = text,
                title            = meta.get("title", ""),
                source           = meta.get("source", ""),
                page_number      = int(meta.get("page_number", 0)),
                dense_score      = round(similarity, 4),
                retrieval_source = "dense",
            ))

        # Sort by similarity descending (ChromaDB already does this,
        # but we sort explicitly to guarantee ordering after dedup)
        chunks.sort(key=lambda c: c.dense_score, reverse=True)
        return chunks

    def _deduplicate(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove duplicate chunk_ids, keeping the first (highest score) occurrence."""
        seen: set[str] = set()
        unique = []
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique.append(chunk)
        if len(unique) < len(chunks):
            logger.warning(
                f"DenseRetriever: deduplicated {len(chunks) - len(unique)} "
                "duplicate chunk(s)"
            )
        return unique