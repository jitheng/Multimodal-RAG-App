import os

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then env vars, then default."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


# OpenAI
OPENAI_API_KEY = _get("OPENAI_API_KEY")
OPENAI_BASE_URL = _get("OPENAI_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1024
GENERATION_MODEL = "gpt-4o"
VISION_MODEL = "gpt-4o"

# Pinecone
PINECONE_API_KEY = _get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = _get("PINECONE_INDEX_NAME", "multimodal-rag")
PINECONE_NAMESPACE = _get("PINECONE_NAMESPACE", "eureka-manuals")
PINECONE_HOST = _get("PINECONE_HOST", "")
PINECONE_METRIC = "cosine"

# Chunking
SEMANTIC_CHUNK_BREAKPOINT = "percentile"
SEMANTIC_CHUNK_THRESHOLD = 85
MAX_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 8
SCORE_THRESHOLD = 0.30

# Paths
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
