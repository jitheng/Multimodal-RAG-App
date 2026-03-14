"""
Embedding utilities.

Provides the OpenAI embedding model and SemanticChunker for text.
All content types (text, tables, images) are embedded via the same model —
tables use markdown representation, images use GPT-4o captions.
"""

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    SEMANTIC_CHUNK_BREAKPOINT,
    SEMANTIC_CHUNK_THRESHOLD,
    get_openai_api_key,
    get_openai_base_url,
)


def get_embeddings_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        openai_api_key=get_openai_api_key(),
        openai_api_base=get_openai_base_url(),
    )


def build_semantic_chunker(embeddings_model: OpenAIEmbeddings) -> SemanticChunker:
    return SemanticChunker(
        embeddings=embeddings_model,
        breakpoint_threshold_type=SEMANTIC_CHUNK_BREAKPOINT,
        breakpoint_threshold_amount=SEMANTIC_CHUNK_THRESHOLD,
    )


def chunk_text_blocks(raw_text_blocks: list, chunker: SemanticChunker) -> list:
    """
    Apply SemanticChunker to each accumulated text block.
    Returns a list of chunk dicts with element_type="text".
    """
    text_chunks = []
    for block in raw_text_blocks:
        doc = Document(
            page_content=block["text"],
            metadata={
                "page_number": block["page_number"],
                "section_title": block["section_title"],
                "source_pdf": block["source_pdf"],
            },
        )
        split_docs = chunker.split_documents([doc])
        for split_doc in split_docs:
            text_chunks.append(
                {
                    "content": split_doc.page_content,
                    "element_type": "text",
                    "page_number": split_doc.metadata.get("page_number", block["page_number"]),
                    "section_title": split_doc.metadata.get("section_title", block["section_title"]),
                    "table_markdown": "",
                    "original_content": split_doc.page_content,
                    "image_path": "",
                    "image_caption": "",
                    "source_pdf": block["source_pdf"],
                }
            )
    return text_chunks
