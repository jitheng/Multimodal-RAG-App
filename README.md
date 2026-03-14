# Multimodal RAG App

A Retrieval-Augmented Generation (RAG) system that ingests PDF technical manuals — including scanned documents with figures, tables, and text — and answers questions with rich, structured responses.

Built with LangChain, OpenAI GPT-4o, and Pinecone.

---

## Features

- **Scanned PDF support** — GPT-4o vision processes each page as an image, extracting text, tables, and figures even when no selectable text exists
- **Multimodal chunking** — separate chunk types for text, tables (markdown), and figures (GPT-4o captions)
- **Dynamic prompts** — response format automatically adapts based on whether retrieved content contains text, tables, images, or a mix
- **Semantic chunking** — text split at natural topic boundaries (not fixed character counts) using LangChain's `SemanticChunker`
- **Pinecone vector store** — fast similarity search with full metadata retrieval
- **Streamlit UI** — chat interface with inline figure display and source attribution per response

---

## Architecture

```
PDF
 │
 ├── Page images (pdf2image)
 │       │
 │       └── GPT-4o Vision ──► text content
 │                          ──► tables (markdown)
 │                          ──► figure descriptions
 │
 ├── Text blocks ──► SemanticChunker ──► text chunks
 ├── Tables ──────────────────────────► table chunks  ──► text-embedding-3-large ──► Pinecone
 └── Figures ─────────────────────────► image chunks
```

**Query flow:**

```
User question
    │
    ▼
text-embedding-3-large  →  Pinecone similarity search (top 8)
    │
    ▼
Classify retrieved chunks (text / table / image)
    │
    ▼
Select dynamic prompt template
    │
    ▼
GPT-4o  →  structured answer (with tables + figure references)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-large (1024 dims) |
| Vector store | Pinecone (Serverless) |
| Framework | LangChain 0.3 (LCEL) |
| PDF parsing | pdf2image + GPT-4o vision |
| Text chunking | LangChain SemanticChunker |
| UI | Streamlit |

---

## Project Structure

```
├── app.py                  # Streamlit chat UI
├── ingest.py               # PDF ingestion pipeline (run once per PDF)
├── rag.py                  # RAG query chain (LCEL)
├── config.py               # All settings and constants
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
└── utils/
    ├── pdf_processor.py    # Page-level GPT-4o vision extraction
    ├── embeddings.py       # OpenAI embeddings + SemanticChunker
    └── pinecone_utils.py   # Pinecone index management and upsert
```

---

## Setup

### Prerequisites

```bash
# macOS (required for pdf2image)
brew install poppler tesseract
```

### Install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1   # or custom endpoint

PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=multimodal-rag
PINECONE_NAMESPACE=eureka-manuals
PINECONE_HOST=                               # optional: direct index host URL
```

> The Pinecone index must be pre-created with **1024 dimensions** and **cosine** metric.

---

## Usage

### 1. Ingest a PDF

```bash
python ingest.py "data/pdfs/your-manual.pdf"
```

This will:
- Render each page to a PNG image
- Analyze each page with GPT-4o vision (extracts text, tables, figures)
- Chunk and embed all content
- Upsert to Pinecone (~2–5 min depending on PDF length)

### 2. Launch the UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 3. Test the RAG chain directly

```bash
python rag.py
```

Runs four sample queries (assembly, troubleshooting, specs, parts list) and prints answers with sources.

---

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub (already done)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo `jitheng/Multimodal-RAG-App`, branch `main`, file `app.py`
4. Click **Advanced settings → Secrets** and paste:

```toml
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "multimodal-rag"
PINECONE_NAMESPACE = "eureka-manuals"
PINECONE_HOST = "your-pinecone-host-url"
```

5. Click **Deploy** — Streamlit Cloud installs dependencies and launches the app.

> System dependency `poppler-utils` is declared in `packages.txt` and installed automatically.

---

## Example Queries

- *"How do I assemble the vacuum cleaner?"*
- *"What parts are included in the box?"*
- *"Why is the suction weak? How do I troubleshoot it?"*
- *"What are the filter specifications?"*

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `OPENAI_BASE_URL` | No | Custom OpenAI endpoint (default: `https://api.openai.com/v1`) |
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `PINECONE_INDEX_NAME` | Yes | Name of the Pinecone index |
| `PINECONE_NAMESPACE` | No | Namespace within the index (default: `eureka-manuals`) |
| `PINECONE_HOST` | No | Direct index host URL (skips index lookup if set) |
