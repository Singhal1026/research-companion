# Research Companion

An AI system that lets you query across multiple research papers using hybrid search (vector + BM25), cross-encoder re-ranking, and a reasoning agent layer. Built with near-zero cost using open-source models.

## What It Does

Drop in research PDFs → ask questions across all of them → get answers with source citations.

## Architecture

```
PDF ingestion → chunking → dual indexing (ChromaDB + BM25)
                                    ↓
              query → hybrid search → re-ranking → LLM → answer + citations
```

See `docs/architecture.md` for the full pipeline diagram.

## Stack

| Component | Library | Cost |
|---|---|---|
| PDF extraction | PyMuPDF | free |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | free, CPU |
| Vector DB | ChromaDB | free, local |
| Keyword search | rank-bm25 | free, local |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | free, CPU |
| LLM | Groq (free tier) / Ollama / OpenAI | switchable |
| Agent | LangGraph | free |
| Evaluation | RAGAS | free |
| MLOps | MLflow + DVC | free |
| Deploy | Hugging Face Spaces | free |

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/research-companion.git
cd research-companion
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER and the matching API key
```

**Quickest start (no GPU, free):** Set `LLM_PROVIDER=groq` and add a free key from [console.groq.com](https://console.groq.com).

**Fully local:** Set `LLM_PROVIDER=ollama`, install [Ollama](https://ollama.com), then run `ollama pull mistral`.

### 4. Verify setup

```bash
python -m src.llm.model_loader
```

Should print: `LLM connection successful.`

### 5. Add papers and run

```bash
# Drop PDFs into data/raw/
# Then:
python main.py
```

## Project Structure

```
research-companion/
├── data/
│   ├── raw/            ← drop PDFs here
│   ├── processed/      ← extracted text (DVC tracked)
│   └── chunks/         ← chunked data (DVC tracked)
├── embeddings/         ← ChromaDB persistent store
├── indexes/            ← BM25 index files
├── src/
│   ├── ingestion/      ← pdf_loader, chunker
│   ├── embeddings/     ← embedder (dense + BM25)
│   ├── retrieval/      ← hybrid search, re-ranker
│   ├── llm/            ← multi-provider model loader
│   ├── agent/          ← LangGraph agent (phase 3)
│   └── app/            ← Gradio UI
├── evaluation/         ← RAGAS eval scripts
├── finetuning/         ← QLoRA training (phase 4)
├── mlops/              ← MLflow + DVC config
├── config.py           ← all settings, reads from .env
└── main.py             ← entry point
```

## Build Phases

- **Phase 1** (Weeks 1–2): Core RAG — ingestion, embeddings, basic retrieval, Gradio UI
- **Phase 2** (Week 3): Hybrid search — BM25 + vector, RRF fusion, cross-encoder re-ranking
- **Phase 3** (Week 4): LangGraph agent with multi-step reasoning
- **Phase 4** (Weeks 5–6): QLoRA fine-tuning + RAGAS evaluation
- **Phase 5** (Week 7): MLflow experiment tracking + DVC versioning
- **Phase 6** (Week 8): Docker + Hugging Face Spaces deployment + CI/CD

## Switching LLM Providers

Edit `.env`:

```bash
# Free, fast, no GPU
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant

# Fully local
LLM_PROVIDER=ollama
LLM_MODEL=mistral

# Best quality (paid)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

No code changes needed — `config.py` handles the rest.
