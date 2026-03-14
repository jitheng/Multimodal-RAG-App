"""
Streamlit UI for the Multimodal RAG System.

Run:
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from config import PDF_DIR, PINECONE_NAMESPACE
from ingest import run_ingestion
from rag import get_rag_chain

st.set_page_config(
    page_title="Technical Manual Assistant",
    page_icon="🔧",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "ingested_pdfs" not in st.session_state:
    st.session_state.ingested_pdfs = []


# ---------------------------------------------------------------------------
# Sidebar — PDF upload & ingestion
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📄 Upload Manual")
    st.caption("Upload a PDF technical manual to ingest into the knowledge base.")

    namespace_input = st.text_input("Pinecone namespace", value=PINECONE_NAMESPACE)

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Ingest PDF", type="primary"):
            os.makedirs(PDF_DIR, exist_ok=True)
            save_path = os.path.join(PDF_DIR, uploaded_file.name)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(
                f"Processing '{uploaded_file.name}'... this may take a few minutes for documents with many images."
            ):
                try:
                    count = run_ingestion(save_path, namespace=namespace_input)
                    st.success(f"Ingested {count} chunks from '{uploaded_file.name}'")
                    if uploaded_file.name not in st.session_state.ingested_pdfs:
                        st.session_state.ingested_pdfs.append(uploaded_file.name)
                    # Reset chain so it picks up new vectors
                    st.session_state.rag_chain = None
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    if st.session_state.ingested_pdfs:
        st.divider()
        st.subheader("Ingested manuals")
        for name in st.session_state.ingested_pdfs:
            st.markdown(f"- {name}")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Lazy-load the RAG chain
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Connecting to knowledge base...")
def load_chain(namespace: str):
    return get_rag_chain(namespace=namespace, return_sources=True)


# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------

st.title("🔧 Technical Manual Assistant")
st.caption(
    "Ask about assembly instructions, troubleshooting, parts lists, specifications, and more."
)

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show inline images for assistant messages that had image sources
        if msg["role"] == "assistant" and msg.get("image_sources"):
            for img_info in msg["image_sources"]:
                img_path = img_info.get("image_path", "")
                caption = img_info.get("caption", "")
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption=caption or Path(img_path).name, use_column_width=True)

        # Source expander
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"View {len(msg['sources'])} source(s)"):
                for src in msg["sources"]:
                    m = src if isinstance(src, dict) else src.metadata
                    etype = m.get("element_type", "text")
                    page = m.get("page_number", "?")
                    section = m.get("section_title", "")
                    pdf = m.get("source_pdf", "")
                    icon = {"text": "📝", "table": "📊", "image": "🖼️"}.get(etype, "📄")
                    st.caption(
                        f"{icon} Page {page}"
                        + (f" | {section}" if section else "")
                        + (f" | {pdf}" if pdf else "")
                    )


# Chat input
if prompt := st.chat_input("Ask about assembly, troubleshooting, parts..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Load chain
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                chain = load_chain(namespace_input)
                result = chain.invoke({"question": prompt})

                answer = result["answer"]
                source_docs = result.get("sources", [])

                st.markdown(answer)

                # Collect image sources for inline display
                image_sources = []
                for doc in source_docs:
                    if doc.metadata.get("element_type") == "image":
                        img_path = doc.metadata.get("image_path", "")
                        caption = doc.metadata.get("image_caption", "")
                        if img_path and os.path.exists(img_path):
                            st.image(
                                img_path,
                                caption=caption or Path(img_path).name,
                                use_column_width=True,
                            )
                            image_sources.append({"image_path": img_path, "caption": caption})

                if source_docs:
                    with st.expander(f"View {len(source_docs)} source(s)"):
                        for doc in source_docs:
                            m = doc.metadata
                            etype = m.get("element_type", "text")
                            page = m.get("page_number", "?")
                            section = m.get("section_title", "")
                            pdf = m.get("source_pdf", "")
                            icon = {"text": "📝", "table": "📊", "image": "🖼️"}.get(etype, "📄")
                            st.caption(
                                f"{icon} Page {page}"
                                + (f" | {section}" if section else "")
                                + (f" | {pdf}" if pdf else "")
                            )

                # Persist to history (store metadata dicts, not Document objects)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": [doc.metadata for doc in source_docs],
                        "image_sources": image_sources,
                    }
                )

            except Exception as e:
                error_msg = f"Error generating response: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
