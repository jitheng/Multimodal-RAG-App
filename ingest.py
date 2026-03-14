"""
PDF Ingestion Pipeline

Usage:
    python ingest.py "data/pdfs/manual.pdf"
    python ingest.py "data/pdfs/manual.pdf" --namespace my-namespace

Flow:
    PDF pages → PNG images → GPT-4o vision (text + tables + figures per page)
    → SemanticChunker (text only) → embed all → Pinecone upsert
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

from config import PINECONE_NAMESPACE
from utils.embeddings import build_semantic_chunker, chunk_text_blocks, get_embeddings_model
from utils.pdf_processor import process_pdf
from utils.pinecone_utils import (
    build_pinecone_records,
    get_or_create_index,
    get_pinecone_client,
    upsert_chunks,
)


def make_chunk_id(source_pdf: str, element_type: str, page_number: int, chunk_index: int) -> str:
    base = f"{source_pdf}__{element_type}__page_{page_number}__chunk_{chunk_index}"
    return hashlib.md5(base.encode()).hexdigest()[:16]


def assign_chunk_ids(chunks: list) -> list:
    """Add deterministic IDs — enables idempotent re-ingestion (upsert = overwrite)."""
    counters: dict = {}
    for chunk in chunks:
        key = (chunk["source_pdf"], chunk["element_type"], chunk["page_number"])
        idx = counters.get(key, 0)
        counters[key] = idx + 1
        chunk["chunk_id"] = make_chunk_id(
            chunk["source_pdf"], chunk["element_type"], chunk["page_number"], idx
        )
    return chunks


def run_ingestion(pdf_path: str, namespace: str = PINECONE_NAMESPACE) -> int:
    """
    Ingest a single PDF into Pinecone.
    Returns the number of chunks upserted.
    """
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_name = Path(pdf_path).name

    # Step 1: Extract all content via GPT-4o vision (handles scanned + native PDFs)
    print(f"\n[1/4] Processing PDF with GPT-4o vision: {pdf_name}")
    raw_chunks = process_pdf(pdf_path, pdf_name)

    text_blocks = [
        {
            "text": c["content"],
            "page_number": c["page_number"],
            "section_title": c["section_title"],
            "source_pdf": c["source_pdf"],
            "_image_path": c.get("image_path", ""),
        }
        for c in raw_chunks if c["element_type"] == "text"
    ]
    non_text_chunks = [c for c in raw_chunks if c["element_type"] != "text"]

    print(
        f"      Raw: {len(text_blocks)} text blocks, "
        f"{sum(1 for c in non_text_chunks if c['element_type']=='table')} tables, "
        f"{sum(1 for c in non_text_chunks if c['element_type']=='image')} figures"
    )

    # Step 2: Semantic-chunk the text blocks
    print("[2/4] Semantic chunking of text")
    embeddings_model = get_embeddings_model()
    chunker = build_semantic_chunker(embeddings_model)
    text_chunks = chunk_text_blocks(text_blocks, chunker)

    # Preserve image_path from original text blocks (page image reference)
    for tc in text_chunks:
        matching = next(
            (b for b in text_blocks if b["page_number"] == tc["page_number"]), None
        )
        if matching:
            tc["image_path"] = matching.get("_image_path", "")

    all_chunks = text_chunks + non_text_chunks
    all_chunks = assign_chunk_ids(all_chunks)
    print(f"      Total chunks to ingest: {len(all_chunks)}")

    # Step 3: Embed
    print("[3/4] Embedding chunks")
    records = build_pinecone_records(all_chunks, embeddings_model)

    # Step 4: Upsert to Pinecone
    print("[4/4] Upserting to Pinecone")
    pc = get_pinecone_client()
    index = get_or_create_index(pc)
    upsert_chunks(index, records, namespace=namespace)

    print(f"\nDone. {len(records)} chunks ingested from '{pdf_name}' into namespace '{namespace}'.")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the multimodal RAG system")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--namespace",
        default=PINECONE_NAMESPACE,
        help=f"Pinecone namespace (default: {PINECONE_NAMESPACE})",
    )
    args = parser.parse_args()

    try:
        run_ingestion(args.pdf_path, args.namespace)
        sys.exit(0)
    except Exception as e:
        print(f"\nError during ingestion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
