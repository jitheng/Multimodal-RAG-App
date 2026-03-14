import os

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    """
    Read config value. Priority: Streamlit secrets → env vars → default.
    Called lazily (inside functions) so st.secrets is always initialized.
    """
    try:
        import streamlit as st
        val = st.secrets.get(key, None)
        if val is not None:
            return str(val).strip()
    except Exception:
        pass
    return os.getenv(key, default) or default


# ── lazy accessors (call these instead of module-level constants) ──────────

def get_openai_api_key() -> str:
    return _get("OPENAI_API_KEY")

def get_openai_base_url() -> str:
    return _get("OPENAI_BASE_URL", "https://api.openai.com/v1")

def get_pinecone_api_key() -> str:
    return _get("PINECONE_API_KEY")

def get_pinecone_index_name() -> str:
    return _get("PINECONE_INDEX_NAME", "multimodal-rag")

def get_pinecone_namespace() -> str:
    return _get("PINECONE_NAMESPACE", "eureka-manuals")

def get_pinecone_host() -> str:
    return _get("PINECONE_HOST", "")


# ── module-level constants (non-secret, safe to set at import time) ────────

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1024
GENERATION_MODEL = "gpt-4o"
VISION_MODEL = "gpt-4o"
PINECONE_METRIC = "cosine"

SEMANTIC_CHUNK_BREAKPOINT = "percentile"
SEMANTIC_CHUNK_THRESHOLD = 85
MAX_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

TOP_K = 8
SCORE_THRESHOLD = 0.30

DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

# ── backwards-compat aliases (for non-Streamlit scripts like ingest.py) ───

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "multimodal-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "eureka-manuals")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")
