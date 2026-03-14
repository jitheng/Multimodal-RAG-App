import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1024
GENERATION_MODEL = "gpt-4o"
VISION_MODEL = "gpt-4o"

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "multimodal-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "eureka-manuals")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")
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
