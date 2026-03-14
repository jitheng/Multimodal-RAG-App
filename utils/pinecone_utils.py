"""
Pinecone utilities.

Handles index creation (Serverless), batch upsert, and LangChain VectorStore wrapper.
Metadata schema per vector:
  element_type, page_number, section_title, content, table_markdown,
  image_path, image_caption, original_content, source_pdf, chunk_index
"""

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from config import (
    EMBEDDING_DIMENSIONS,
    PINECONE_API_KEY,
    PINECONE_HOST,
    PINECONE_INDEX_NAME,
    PINECONE_METRIC,
)

_METADATA_CONTENT_LIMIT = 38000  # bytes; Pinecone hard limit is 40KB per field


def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=PINECONE_API_KEY)


def get_or_create_index(pc: Pinecone, index_name: str = PINECONE_INDEX_NAME):
    # If a direct host is configured, connect to it without listing/creating indexes
    if PINECONE_HOST:
        print(f"Connecting to Pinecone index via host: {PINECONE_HOST}")
        return pc.Index(host=PINECONE_HOST)

    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSIONS,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created Pinecone index: {index_name}")
    else:
        print(f"Using existing Pinecone index: {index_name}")
    return pc.Index(index_name)


def _truncate(value: str, limit: int = _METADATA_CONTENT_LIMIT) -> str:
    """Truncate string to byte limit to stay within Pinecone metadata constraints."""
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="ignore") + "...[truncated]"


def build_pinecone_records(chunks: list, embeddings_model: OpenAIEmbeddings) -> list:
    """
    Embed all chunks and build Pinecone-ready records.
    Returns list of (id, values, metadata) tuples.
    """
    records = []
    texts = [c["content"] for c in chunks]

    print(f"  Embedding {len(texts)} chunks...")
    # Batch embed for efficiency
    vectors = embeddings_model.embed_documents(texts)

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        metadata = {
            "element_type": chunk.get("element_type", "text"),
            "page_number": chunk.get("page_number", 0),
            "section_title": chunk.get("section_title", ""),
            "content": _truncate(chunk.get("content", "")),
            "table_markdown": _truncate(chunk.get("table_markdown", "")),
            "image_path": chunk.get("image_path", ""),
            "image_caption": _truncate(chunk.get("image_caption", "")),
            "original_content": _truncate(chunk.get("original_content", "")),
            "source_pdf": chunk.get("source_pdf", ""),
            "chunk_index": i,
        }
        records.append({"id": chunk_id, "values": vector, "metadata": metadata})

    return records


def upsert_chunks(index, records: list, namespace: str, batch_size: int = 100):
    """Upsert records to Pinecone in batches."""
    total = len(records)
    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"  Upserted {min(i + batch_size, total)}/{total} records")


def get_vectorstore(
    index, embeddings_model: OpenAIEmbeddings, namespace: str
) -> PineconeVectorStore:
    return PineconeVectorStore(
        index=index,
        embedding=embeddings_model,
        namespace=namespace,
        text_key="content",
    )
