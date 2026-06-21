"""
src/llm/model_loader.py

Single function get_llm() returns a LangChain-compatible chat model
based on LLM_PROVIDER in your .env. Switch providers by changing one line.

Supported:
    groq   — free API, fast, no GPU  (default)
    ollama — fully local, no API key
    openai — paid, best quality
"""
from __future__ import annotations
from langchain_core.language_models.chat_models import BaseChatModel
import config


def get_llm() -> BaseChatModel:
    """
    Return a LangChain chat model for the configured provider.

    Usage:
        from src.llm.model_loader import get_llm
        llm = get_llm()
        response = llm.invoke("Hello!")
    """
    provider = config.LLM_PROVIDER
    model    = config.LLM_MODEL

    if provider == "groq":
        return _groq(model)
    elif provider == "ollama":
        return _ollama(model)
    elif provider == "openai":
        return _openai(model)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Set LLM_PROVIDER to one of: groq, ollama, openai"
        )


# ── Provider implementations ──────────────────────────────────────────────────

def _groq(model: str) -> BaseChatModel:
    """Groq: free tier, fast inference via cloud. No GPU needed."""
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("Run: pip install langchain-groq")

    if not config.GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY missing. Get a free key at https://console.groq.com"
        )

    return ChatGroq(
        model=model,
        api_key=config.GROQ_API_KEY,
        temperature=0.1,     # low temp = more factual, less creative
        max_tokens=1024,
    )


def _ollama(model: str) -> BaseChatModel:
    """Ollama: fully local. Install from https://ollama.com then run:
        ollama pull mistral   (or whichever model you set in .env)
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError("Run: pip install langchain-ollama")

    return ChatOllama(
        model=model,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.1,
    )


def _openai(model: str) -> BaseChatModel:
    """OpenAI: paid API, highest quality. gpt-4o-mini is cheapest."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("Run: pip install langchain-openai")

    if not config.OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY missing.")

    return ChatOpenAI(
        model=model,
        api_key=config.OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=1024,
    )


# ── Quick test (run this file directly to verify your setup) ──────────────────
if __name__ == "__main__":
    import config as cfg
    cfg.validate()
    print(f"Provider : {cfg.LLM_PROVIDER}")
    print(f"Model    : {cfg.LLM_MODEL}")
    print("Loading model...")
    llm = get_llm()
    response = llm.invoke("Reply with exactly: 'LLM connection successful.'")
    print(f"Response : {response.content}")
