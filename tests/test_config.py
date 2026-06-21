"""
tests/test_config.py — basic sanity checks for project config.
More tests added as each module is built.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_imports():
    import config
    assert config.CHUNK_SIZE == 500
    assert config.CHUNK_OVERLAP == 100
    assert config.RETRIEVAL_TOP_K == 20
    assert config.RERANK_TOP_K == 4


def test_supported_providers():
    import config
    assert "groq" in config.SUPPORTED_PROVIDERS
    assert "ollama" in config.SUPPORTED_PROVIDERS
    assert "openai" in config.SUPPORTED_PROVIDERS


def test_paths_exist():
    import config
    assert config.ROOT.exists()
    assert config.DATA_RAW.exists()
    assert config.EMBEDDINGS_DIR.exists()
    assert config.INDEXES_DIR.exists()
