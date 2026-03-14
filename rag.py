"""
RAG Query Pipeline

Implements dynamic prompt selection based on retrieved content types,
formatted context construction, and an LCEL chain for answer generation.

Standalone test:
    python rag.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from config import (
    GENERATION_MODEL,
    SCORE_THRESHOLD,
    TOP_K,
    get_openai_api_key,
    get_openai_base_url,
    get_pinecone_namespace,
)
from utils.embeddings import get_embeddings_model
from utils.pinecone_utils import get_or_create_index, get_pinecone_client, get_vectorstore

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_BASE_SYSTEM = (
    "You are a technical support assistant for product manuals. "
    "Answer based ONLY on the provided context. "
    "If the information is not in the context, say so clearly. "
    "Be precise about part names, steps, and safety warnings."
)

TEXT_ONLY_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", _BASE_SYSTEM),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\n"
            "Provide a clear, step-by-step answer if applicable. "
            "Cite page numbers when referencing specific steps.",
        ),
    ]
)

TEXT_TABLE_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", _BASE_SYSTEM),
        (
            "human",
            "Context (includes tables):\n{context}\n\nQuestion: {question}\n\n"
            "Instructions:\n"
            "- When referencing table data, reproduce the relevant rows in markdown table format.\n"
            "- For specifications or part lists, preserve the tabular structure in your answer.\n"
            "- Cite page numbers and table context.",
        ),
    ]
)

TEXT_IMAGE_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", _BASE_SYSTEM),
        (
            "human",
            "Context (includes figures/diagrams):\n{context}\n\nQuestion: {question}\n\n"
            "Instructions:\n"
            "- Reference figures as: [Figure: page X — brief description].\n"
            "- Describe spatial relationships (e.g., 'Component A attaches to slot B').\n"
            "- If a figure shows assembly steps, enumerate them based on the figure description.\n"
            "- Note any safety callouts visible in the figures.",
        ),
    ]
)

FULL_MULTIMODAL_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", _BASE_SYSTEM),
        (
            "human",
            "Context (includes text, tables, and figures):\n{context}\n\nQuestion: {question}\n\n"
            "Instructions:\n"
            "- Structure your response: direct answer, then relevant table data (markdown), "
            "then figure references.\n"
            "- Use markdown formatting for tables.\n"
            "- Reference figures as: [Figure: page X — description].\n"
            "- Cite specific steps or part numbers from tables.\n"
            "- For assembly or troubleshooting, number the steps.",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------


def classify_retrieved_docs(docs: list) -> dict:
    return {
        "has_text": any(d.metadata.get("element_type") == "text" for d in docs),
        "has_tables": any(d.metadata.get("element_type") == "table" for d in docs),
        "has_images": any(d.metadata.get("element_type") == "image" for d in docs),
    }


def select_prompt(content_types: dict) -> ChatPromptTemplate:
    if content_types["has_tables"] and content_types["has_images"]:
        return FULL_MULTIMODAL_TEMPLATE
    if content_types["has_tables"]:
        return TEXT_TABLE_TEMPLATE
    if content_types["has_images"]:
        return TEXT_IMAGE_TEMPLATE
    return TEXT_ONLY_TEMPLATE


def format_context(docs: list) -> str:
    sections = []
    for doc in docs:
        meta = doc.metadata
        etype = meta.get("element_type", "text")
        page = meta.get("page_number", "?")
        section = meta.get("section_title", "")
        header = f"[Page {page}" + (f" | {section}" if section else "") + f" | Type: {etype}]"

        if etype == "table":
            body = meta.get("table_markdown") or doc.page_content
            sections.append(f"{header}\n{body}")
        elif etype == "image":
            caption = meta.get("image_caption") or doc.page_content
            img_path = meta.get("image_path", "")
            ref = f"\nFigure path: {img_path}" if img_path else ""
            sections.append(f"{header}\nFigure description: {caption}{ref}")
        else:
            sections.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------


def build_rag_chain(vectorstore, return_sources: bool = False):
    """
    Build the LCEL RAG chain.

    If return_sources=False (default): chain returns answer string.
    If return_sources=True: chain returns dict {"answer": str, "sources": list[Document]}.
    """
    llm = ChatOpenAI(
        model=GENERATION_MODEL,
        temperature=0.1,
        openai_api_key=get_openai_api_key(),
        openai_api_base=get_openai_base_url(),
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD},
    )

    def retrieve_and_prepare(inputs: dict) -> dict:
        question = inputs["question"]
        docs = retriever.invoke(question)
        content_types = classify_retrieved_docs(docs)
        prompt_template = select_prompt(content_types)
        context_str = format_context(docs)
        return {
            "prompt_template": prompt_template,
            "context": context_str,
            "question": question,
            "source_docs": docs,
            "content_types": content_types,
        }

    def apply_prompt(prepared: dict):
        return prepared["prompt_template"].invoke(
            {"context": prepared["context"], "question": prepared["question"]}
        )

    if return_sources:
        # Return both answer and source docs
        def run_with_sources(inputs: dict) -> dict:
            prepared = retrieve_and_prepare(inputs)
            messages = apply_prompt(prepared)
            answer = (llm | StrOutputParser()).invoke(messages)
            return {"answer": answer, "sources": prepared["source_docs"]}

        return RunnableLambda(run_with_sources)

    chain = (
        RunnablePassthrough()
        | RunnableLambda(retrieve_and_prepare)
        | RunnableLambda(apply_prompt)
        | llm
        | StrOutputParser()
    )
    return chain


def get_rag_chain(namespace: str = None, return_sources: bool = True):
    namespace = namespace or get_pinecone_namespace()
    """Convenience factory: initialise all dependencies and return the chain."""
    embeddings_model = get_embeddings_model()
    pc = get_pinecone_client()
    index = get_or_create_index(pc)
    vectorstore = get_vectorstore(index, embeddings_model, namespace)
    return build_rag_chain(vectorstore, return_sources=return_sources)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "How do I assemble the vacuum cleaner?",
        "What are the filter specifications?",
        "Why is the suction weak? How do I troubleshoot it?",
        "What parts are included in the box?",
    ]

    print("Initialising RAG chain...")
    chain = get_rag_chain(return_sources=True)

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = chain.invoke({"question": q})
        print(f"\nA: {result['answer']}")
        print(f"\nSources ({len(result['sources'])} chunks):")
        for src in result["sources"]:
            m = src.metadata
            print(f"  - Page {m.get('page_number')} | {m.get('element_type')} | {m.get('section_title', '')}")
