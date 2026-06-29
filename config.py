"""
config.py — central settings loaded from .env
All other modules import from here. Never read os.environ directly elsewhere.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_CHUNKS    = ROOT / "data" / "chunks"
EMBEDDINGS_DIR = ROOT / "embeddings"
INDEXES_DIR    = ROOT / "indexes"
MODELS_DIR     = ROOT / "models"

# Ensure dirs exist at import time (safe to call repeatedly)
for _p in [DATA_RAW, DATA_PROCESSED, DATA_CHUNKS, EMBEDDINGS_DIR, INDEXES_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ── LLM provider ─────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL:    str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

GROQ_API_KEY:    str = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY:  str = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SUPPORTED_PROVIDERS = ("groq", "ollama", "openai")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Batch size for sentence-transformers encode() calls.
# 64 is a safe default for CPU. Raise to 128-256 if you have a GPU.
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# Path where the BM25 index is serialised to disk.
BM25_INDEX_PATH: str = str(INDEXES_DIR / "bm25.pkl")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE:    int = 500
CHUNK_OVERLAP: int = 100

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RERANK_TOP_K:    int = int(os.getenv("RERANK_TOP_K", "4"))

# Cross-encoder model for re-ranking (runs locally)
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Chroma collection name ────────────────────────────────────────────────────
CHROMA_COLLECTION: str = "research_papers"

# ── Router ────────────────────────────────────────────────────────────────────
# Keywords that short-circuit the LLM classification call (faster, cheaper).
# If a query matches one of these patterns, the route is decided immediately
# without spending an API call on classification.
ROUTER_AGENT_KEYWORDS:  tuple = ("compare", "difference between", "contrast",
                                  "versus", "vs ", "how do .* differ")
ROUTER_DIRECT_KEYWORDS: tuple = ("what is", "what are", "define ", "explain ",
                                  "who invented", "when was")

# ── Validation ────────────────────────────────────────────────────────────────
def validate() -> None:
    """Call at app startup to catch missing config early."""
    if LLM_PROVIDER not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER='{LLM_PROVIDER}' not supported. "
            f"Choose from: {SUPPORTED_PROVIDERS}"
        )
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set.")