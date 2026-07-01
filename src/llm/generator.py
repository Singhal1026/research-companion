"""
src/llm/generator.py

Assembles retrieved chunks into a context window, builds a citation-aware
prompt, calls the LLM, and returns a structured GeneratorResponse.

DESIGN DECISIONS:

  Structured GeneratorResponse
    Returns a dataclass, not a raw string. The UI, evaluator, and agent
    all need: answer text, citations, token usage, latency, route used.
    Returning a raw string forces callers to reconstruct this themselves.

  Citation-aware prompting
    The prompt numbers each chunk [1], [2], [3]... and explicitly
    instructs the LLM to cite sources inline. Without explicit instruction,
    LLMs tend to ignore retrieved context and answer from training data —
    which defeats the purpose of RAG and makes faithfulness scores low.

  Route-aware prompt templates
    DIRECT queries have no retrieved context — the prompt is just a clean
    question. HYBRID/SINGLE_PAPER queries use a context+citations template.
    Using the same prompt for both wastes tokens and confuses the model
    ("Based on the following context: [nothing] ...").

  Context token budget
    Each chunk is ~500 tokens. With 4 chunks + system prompt + question,
    you're at ~2500 tokens before the answer. We enforce a hard budget
    (CONTEXT_TOKEN_BUDGET) and truncate the context rather than silently
    overflowing the model's context window. Overflow causes silent
    truncation at the API level — you lose the most important context
    (which is usually at the start).

  Token counting without a tokeniser
    Proper token counting requires loading the model's tokeniser (slow,
    heavy). We use chars / 4 as an approximation (1 token ≈ 4 chars for
    English text). This is accurate enough for budget management — the
    real tokeniser would give ~5% different results.

  Streaming
    generate_stream() yields tokens as they arrive via LangChain's
    stream() interface. Used by the Gradio UI for real-time display.
    generate() collects the full response for batch use and evaluation.

  Empty retrieval handling
    If the retrieval pipeline returned no chunks (empty collection or
    DIRECT route), the generator returns a clear answer rather than
    hallucinating context. For DIRECT: answers from training data.
    For empty retrieval on HYBRID: tells the user nothing was found.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

import config
from src.llm.model_loader import get_llm
from src.retrieval.result import RetrievedChunk
from src.retrieval.router import Route, RouterResult


# ── GeneratorResponse ─────────────────────────────────────────────────────────

@dataclass
class GeneratorResponse:
    """
    Structured output from the generator.

    Attributes:
        answer:          The LLM's answer text.
        citations:       List of citation strings, one per retrieved chunk.
                         Format: "attention_paper.pdf — page 4"
                         Index matches [1], [2] markers in the answer.
        chunks_used:     The RetrievedChunks that were passed to the LLM.
        route:           Which routing strategy was used.
        query:           The original user query.
        input_tokens:    Approximate token count of the full prompt.
        output_tokens:   Approximate token count of the answer.
        latency_ms:      Total generation time in milliseconds.
        was_truncated:   True if context was truncated to fit token budget.
    """
    answer:        str
    citations:     list[str]
    chunks_used:   list[RetrievedChunk]
    route:         Route
    query:         str
    input_tokens:  int  = 0
    output_tokens: int  = 0
    latency_ms:    int  = 0
    was_truncated: bool = False

    @property
    def has_context(self) -> bool:
        """True if the answer was grounded in retrieved chunks."""
        return len(self.chunks_used) > 0

    def formatted_citations(self) -> str:
        """
        Citation block ready for the UI.
        Example:
            Sources:
            [1] attention_paper.pdf — page 4
            [2] bert_paper.pdf — page 7
        """
        if not self.citations:
            return ""
        lines = ["**Sources:**"]
        for i, cit in enumerate(self.citations, start=1):
            lines.append(f"[{i}] {cit}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GeneratorResponse("
            f"route={self.route.value}, "
            f"chunks={len(self.chunks_used)}, "
            f"tokens={self.input_tokens}+{self.output_tokens}, "
            f"latency={self.latency_ms}ms, "
            f"truncated={self.was_truncated})"
        )


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator:
    """
    Assembles context from retrieved chunks and generates an LLM answer.

    Args:
        token_budget: Max tokens to use for retrieved context.
                      Defaults to config.CONTEXT_TOKEN_BUDGET.

    Usage:
        generator = Generator()

        # Batch (returns full response)
        response = generator.generate(query, chunks, route_result)
        print(response.answer)
        print(response.formatted_citations())

        # Streaming (yields tokens for Gradio)
        for token in generator.generate_stream(query, chunks, route_result):
            print(token, end="", flush=True)
    """

    def __init__(self, token_budget: int = config.CONTEXT_TOKEN_BUDGET):
        self._token_budget = token_budget
        self._llm          = get_llm()

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        query:        str,
        chunks:       list[RetrievedChunk],
        route_result: RouterResult,
    ) -> GeneratorResponse:
        """
        Generate a full answer. Blocks until the LLM finishes.

        Args:
            query:        User query string.
            chunks:       Output of RetrievalPipeline.retrieve().
            route_result: Output of QueryRouter.classify().

        Returns:
            GeneratorResponse with answer, citations, and usage metadata.
        """
        start_ms = int(time.time() * 1000)

        context, citations, was_truncated = self._build_context(chunks)
        messages = self._build_messages(query, context, route_result.route)
        prompt_text = " ".join(m.content for m in messages)

        try:
            response     = self._llm.invoke(messages)
            answer       = response.content.strip()
        except Exception as e:
            logger.error(f"Generator: LLM call failed: {e}")
            answer = "I encountered an error generating a response. Please try again."

        latency_ms = int(time.time() * 1000) - start_ms

        return GeneratorResponse(
            answer        = answer,
            citations     = citations,
            chunks_used   = chunks,
            route         = route_result.route,
            query         = query,
            input_tokens  = self._count_tokens(prompt_text),
            output_tokens = self._count_tokens(answer),
            latency_ms    = latency_ms,
            was_truncated = was_truncated,
        )

    def generate_stream(
        self,
        query:        str,
        chunks:       list[RetrievedChunk],
        route_result: RouterResult,
    ) -> Iterator[str]:
        """
        Stream tokens as they arrive. Used by Gradio for real-time display.

        Yields individual token strings. The caller accumulates them.
        Citations are NOT streamed — they're appended after generation ends.

        Usage (Gradio):
            for token in generator.generate_stream(query, chunks, route):
                accumulated += token
                yield accumulated   # Gradio needs the full string so far
        """
        context, _, _ = self._build_context(chunks)
        messages      = self._build_messages(query, context, route_result.route)

        try:
            for chunk in self._llm.stream(messages):
                # LangChain stream yields AIMessageChunk objects
                yield chunk.content
        except Exception as e:
            logger.error(f"Generator: stream failed: {e}")
            yield "\n\n[Error: generation failed. Please try again.]"

    # ── Context assembly ──────────────────────────────────────────────────────

    def _build_context(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, list[str], bool]:
        """
        Assemble chunks into a numbered context block within the token budget.

        Returns:
            context:      Formatted context string with [1], [2] markers.
            citations:    List of citation strings parallel to chunk numbers.
            was_truncated: True if any chunks were dropped due to budget.

        Token budget logic:
            We track approximate tokens consumed as we add chunks.
            If adding the next chunk would exceed the budget, we stop.
            Chunks are already sorted by rerank_score (best first), so
            we always include the highest-quality chunks and drop the rest.
        """
        if not chunks:
            return "", [], False

        context_parts: list[str] = []
        citations:     list[str] = []
        tokens_used = 0
        was_truncated = False

        for i, chunk in enumerate(chunks, start=1):
            chunk_tokens = self._count_tokens(chunk.text)

            if tokens_used + chunk_tokens > self._token_budget:
                was_truncated = True
                logger.debug(
                    f"Generator: context truncated at chunk {i} "
                    f"({tokens_used} tokens used, budget={self._token_budget})"
                )
                break

            # Format chunk with its citation number
            context_parts.append(f"[{i}] {chunk.text}")
            citations.append(chunk.citation())
            tokens_used += chunk_tokens

        context = "\n\n".join(context_parts)
        return context, citations, was_truncated

    # ── Prompt templates ──────────────────────────────────────────────────────

    def _build_messages(
        self,
        query:   str,
        context: str,
        route:   Route,
    ) -> list[SystemMessage | HumanMessage]:
        """
        Build LangChain messages list based on route and context availability.

        Three templates:
          1. DIRECT (no context)     — clean Q&A, no citation instruction
          2. With context            — citation-aware RAG template
          3. Empty retrieval fallback — tells user nothing was found
        """
        if route == Route.DIRECT:
            return self._direct_messages(query)

        if not context:
            return self._no_context_messages(query)

        return self._rag_messages(query, context)

    def _rag_messages(
        self,
        query:   str,
        context: str,
    ) -> list[SystemMessage | HumanMessage]:
        """
        Main RAG template — used for HYBRID and SINGLE_PAPER routes.

        The system prompt:
          - Sets the role as a research assistant
          - Explains the citation format ([1], [2], etc.)
          - Explicitly forbids answering from outside the provided context
            (this is the key anti-hallucination instruction)
          - Asks for a structured, concise answer
        """
        system = SystemMessage(content=(
            "You are a precise research assistant. "
            "You answer questions strictly based on the provided research paper excerpts. "
            "Each excerpt is numbered [1], [2], etc. "
            "When you use information from an excerpt, cite it inline like this: "
            "\"Transformers use self-attention [1] which allows parallel computation [2].\" "
            "If the answer cannot be found in the provided excerpts, say: "
            "\"The provided papers do not contain enough information to answer this question.\" "
            "Do not use knowledge from outside the provided excerpts. "
            "Be concise and precise. Avoid bullet points unless the question asks for a list."
        ))

        human = HumanMessage(content=(
            f"Research paper excerpts:\n\n"
            f"{context}\n\n"
            f"Question: {query}"
        ))

        return [system, human]

    def _direct_messages(self, query: str) -> list[SystemMessage | HumanMessage]:
        """
        Template for DIRECT route — general knowledge, no retrieved context.
        No citation instruction because there are no sources to cite.
        """
        system = SystemMessage(content=(
            "You are a knowledgeable research assistant. "
            "Answer the question clearly and concisely. "
            "If you are uncertain, say so."
        ))
        human  = HumanMessage(content=query)
        return [system, human]

    def _no_context_messages(self, query: str) -> list[SystemMessage | HumanMessage]:
        """
        Fallback when retrieval returned nothing — tells user clearly.
        Better than letting the LLM hallucinate an answer.
        """
        system = SystemMessage(content=(
            "You are a research assistant. "
            "You only answer based on indexed research papers."
        ))
        human  = HumanMessage(content=(
            f"Question: {query}\n\n"
            "Note: No relevant excerpts were found in the indexed papers. "
            "Please tell the user that you could not find relevant information "
            "in the currently indexed documents, and suggest they check if the "
            "relevant papers have been added to the system."
        ))
        return [system, human]

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _count_tokens(text: str) -> int:
        """
        Approximate token count. 1 token ≈ 4 chars for English text.
        Fast and good enough for budget management.
        Use a real tokeniser (tiktoken) if you need exact counts.
        """
        return max(1, len(text) // config.CHARS_PER_TOKEN)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from src.retrieval.router import QueryRouter

    query = sys.argv[1] if len(sys.argv) > 1 else "What is attention in transformers?"

    # Minimal test without real retrieval — just verifies LLM connection
    import config as cfg
    cfg.validate()

    router       = QueryRouter()
    route_result = router.classify(query)
    generator    = Generator()

    print(f"\nQuery : {query}")
    print(f"Route : {route_result.route.value}")
    print(f"\nGenerating...\n")

    response = generator.generate(query, [], route_result)

    print(f"Answer:\n{response.answer}")
    print(f"\n{response.formatted_citations()}" if response.citations else "")
    print(f"\n── Metadata ───────────────────────────────")
    print(f"Tokens  : {response.input_tokens} in / {response.output_tokens} out")
    print(f"Latency : {response.latency_ms}ms")
    print(f"Route   : {response.route.value}")