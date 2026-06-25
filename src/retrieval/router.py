"""
src/retrieval/router.py

The router sits between the user's query and the retrieval pipeline.
It classifies query intent and returns a Route, which tells the rest
of the system which strategy to use.

WHY THIS EXISTS:
  Not every query needs the same treatment. Without a router, every
  query — even "what is BERT?" — goes through your full hybrid search
  + re-ranking pipeline. That's slow and often gives worse answers
  (retrieved chunks about BERT don't help answer a definition question
  the LLM already knows). The router matches the query to the right
  strategy before anything expensive runs.

THE FOUR ROUTES:

  DIRECT      — General knowledge question. The LLM can answer from
                training data. Retrieval would add noise, not signal.
                Example: "What is attention in transformers?"

  SINGLE_PAPER — User is asking about one specific named paper.
                Retrieval is scoped to that document only.
                Example: "Summarise the BERT paper"
                Example: "What dataset did the GPT-3 paper use?"

  HYBRID      — Cross-paper search. The user wants knowledge spread
                across multiple documents. Full vector + BM25 pipeline.
                Example: "What techniques exist for positional encoding?"
                Example: "Find papers that discuss contrastive learning"

  AGENT       — Multi-hop reasoning. Needs multiple retrievals, synthesis,
                comparison across papers. Handed to LangGraph agent.
                Example: "Compare self-attention in the original transformer
                          paper vs BERT"
                Example: "What did these three papers conclude differently
                          about dropout?"

HOW CLASSIFICATION WORKS (two-stage):

  Stage 1 — Keyword heuristics (free, instant):
    Before calling the LLM at all, check the query against known
    keyword patterns. "compare X and Y" is almost always AGENT.
    "what is X" with no paper name is almost always DIRECT.
    If a keyword matches → return immediately, no API call needed.

  Stage 2 — LLM classification (one cheap fast call):
    If keywords don't resolve it, ask the LLM to classify.
    Uses a small fast model (llama-3.1-8b-instant on Groq is near-instant).
    Returns one of the four route labels.

STATUS: STUB
  Classification logic is not implemented yet — classify() returns HYBRID
  for everything so the pipeline works end-to-end during Phase 1–2.
  Full implementation happens in Phase 3 when the agent exists to route to.
  The data structures and interface are final — no refactoring needed later.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


# ── Route enum ────────────────────────────────────────────────────────────────

class Route(str, Enum):
    """
    The four routing strategies.

    Inherits from str so a Route can be compared directly to a string:
        route == "direct"   # works
        route == Route.DIRECT  # also works
    This makes it easy to use in if/match statements.
    """
    DIRECT       = "direct"
    SINGLE_PAPER = "single_paper"
    HYBRID       = "hybrid"
    AGENT        = "agent"


# ── RouterResult ──────────────────────────────────────────────────────────────

@dataclass
class RouterResult:
    """
    Everything the rest of the pipeline needs from the router.

    Fields:
        route:        Which strategy to use (see Route enum above).
        query:        The original user query, unchanged.
        target_paper: Only set when route == SINGLE_PAPER.
                      The paper title/name the user asked about.
                      The retriever uses this to filter chunks to one doc.
        reasoning:    How the router made its decision. Useful for debugging
                      and for showing in the UI ("I'm searching across all
                      papers because your query spans multiple documents").
    """
    route:        Route
    query:        str
    target_paper: str | None = None
    reasoning:    str        = ""


# ── Router ────────────────────────────────────────────────────────────────────

class QueryRouter:
    """
    Classifies a user query into a Route.

    Usage (once implemented):
        router = QueryRouter()
        result = router.classify("Compare BERT and GPT-2 architectures")
        # result.route == Route.AGENT
        # result.reasoning == "Query contains 'compare' — multi-hop reasoning needed"

    Currently returns Route.HYBRID for everything (stub behaviour).
    """

    def classify(self, query: str) -> RouterResult:
        """
        Classify a query into a Route.

        Args:
            query: The raw user query string.

        Returns:
            RouterResult with route, optional target_paper, and reasoning.
        """
        query = query.strip()

        # ── Stage 1: keyword heuristics ───────────────────────────────────────
        # Check cheap patterns before spending an API call.

        keyword_result = self._keyword_classify(query)
        if keyword_result is not None:
            logger.debug(f"Router (keyword): '{query[:50]}' → {keyword_result.route}")
            return keyword_result

        # ── Stage 2: LLM classification ───────────────────────────────────────
        # TODO (Phase 3): call LLM to classify queries that keywords didn't
        # resolve. Stub falls through to default.

        # ── Stub default ──────────────────────────────────────────────────────
        # During Phase 1–2, everything routes to HYBRID so the pipeline runs
        # end-to-end. Replace this when Phase 3 agent is built.
        logger.debug(f"Router (stub default): '{query[:50]}' → HYBRID")
        return RouterResult(
            route     = Route.HYBRID,
            query     = query,
            reasoning = "Stub: defaulting to hybrid search (Phase 3 will classify properly)",
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _keyword_classify(self, query: str) -> RouterResult | None:
        """
        Fast pattern matching before the LLM call.
        Returns None if no pattern matched (fall through to LLM stage).

        Why this order matters:
          Agent keywords are checked BEFORE direct keywords because a query
          like "compare what is X vs what is Y" contains both "compare" (agent)
          and "what is" (direct). The more specific, expensive route wins.
        """
        q = query.lower()

        # ── Agent patterns (multi-hop reasoning) ──────────────────────────────
        # These almost always need the agent. Check first — they're more specific.
        agent_patterns = [
            r"\bcompare\b",
            r"\bdifferences? between\b",   # covers both "difference" and "differences"
            r"\bsimilarities? between\b",   # covers both "similarity" and "similarities"
            r"\bcontrast\b",
            r"\bversus\b",
            r"\bvs\.?\b",
            r"\bhow do .{1,40} differ\b",
            r"\bwhat did .{1,40} conclude differently\b",
        ]
        for pattern in agent_patterns:
            if re.search(pattern, q):
                return RouterResult(
                    route     = Route.AGENT,
                    query     = query,
                    reasoning = f"Keyword match '{pattern}' → multi-hop reasoning needed",
                )

        # ── Direct patterns (general knowledge, no retrieval) ─────────────────
        # Only fire if the query doesn't name a paper (checked via simple
        # heuristic: no quoted text, no "paper", no author-like patterns).
        # If the user says "what is BERT?" that's direct.
        # If they say "what is the BERT paper's architecture?" that's hybrid.
        direct_patterns = [
            r"^what is\b",
            r"^what are\b(?! the differences?\b)(?! the similarities?\b)",
            r"^define\b",
            r"^who invented\b",
            r"^when was\b",
            r"^explain\b",
        ]
        paper_indicators = ["paper", "study", "article", "research", "authors", '"', "'"]
        has_paper_ref = any(ind in q for ind in paper_indicators)

        if not has_paper_ref:
            for pattern in direct_patterns:
                if re.search(pattern, q):
                    return RouterResult(
                        route     = Route.DIRECT,
                        query     = query,
                        reasoning = f"General knowledge question ('{pattern}', no paper reference) → skip retrieval",
                    )

        # No keyword matched — let the LLM decide (or stub default)
        return None


# ── Convenience function ──────────────────────────────────────────────────────

# A module-level instance so callers can do:
#   from src.retrieval.router import classify
#   result = classify("compare BERT and GPT-2")
# instead of instantiating QueryRouter() themselves.
_router = QueryRouter()

def classify(query: str) -> RouterResult:
    """Module-level shortcut. See QueryRouter.classify() for full docs."""
    return _router.classify(query)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        ("What is attention in transformers?",               Route.DIRECT),
        ("Compare BERT vs GPT-2 architectures",              Route.AGENT),
        ("What are the differences between BERT and RoBERTa?", Route.AGENT),
        ("Find papers on contrastive learning",              Route.HYBRID),  # stub
        ("Summarise the attention is all you need paper",    Route.HYBRID),  # stub (Phase 3)
    ]

    router = QueryRouter()
    all_passed = True

    print("\nRouter test\n" + "─" * 50)
    for query, expected in test_queries:
        result = router.classify(query)
        status = "✓" if result.route == expected else "✗"
        if result.route != expected:
            all_passed = False
        print(f"  {status} [{result.route.value:<12}] {query}")
        print(f"       reasoning: {result.reasoning}")

    print()
    if all_passed:
        print("All tests passed.")
    else:
        print("Some tests failed — expected routes are approximate (stub returns HYBRID for unmatched queries).")
