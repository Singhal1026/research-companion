"""
main.py — entry point for the Research Companion system.

Currently just validates config. Will wire up the full pipeline
as each phase is built.
"""
import config

def main():
    config.validate()
    print(f"Research Companion starting...")
    print(f"  LLM provider : {config.LLM_PROVIDER}")
    print(f"  LLM model    : {config.LLM_MODEL}")
    print(f"  Embedding    : {config.EMBEDDING_MODEL}")
    print(f"  Retrieval k  : {config.RETRIEVAL_TOP_K} → rerank to {config.RERANK_TOP_K}")
    print("\nSystem ready. UI coming in Phase 1.")

if __name__ == "__main__":
    main()
