from __future__ import annotations

import os
import streamlit as st
from dotenv import load_dotenv

from .ingest import _split_into_docs
from .memory import build_memory
from .chains import answer_with_guard
from .vectorstore import build_faiss_from_docs, load_local_faiss, save_local_faiss


def ensure_api_key():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Create .env and set the key.")
        st.stop()


def get_state():
    if "vs" not in st.session_state:
        st.session_state.vs = None
    if "memory" not in st.session_state:
        st.session_state.memory = build_memory()
    return st.session_state


def sidebar_index_controls(index_dir: str):
    st.sidebar.header("Index Notes")
    title = st.sidebar.text_input("Title", value="Pasted Notes")
    pasted = st.sidebar.text_area("Paste notes here", height=180)
    uploaded = st.sidebar.file_uploader("Or upload a local .md/.txt file", type=["md", "txt"])
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Index Pasted"):
            if not pasted.strip():
                st.sidebar.warning("Paste some text first.")
            else:
                docs = _split_into_docs(pasted, title=title)
                vs = build_faiss_from_docs(docs)
                save_local_faiss(vs, index_dir)
                st.session_state.vs = vs
                st.sidebar.success(f"Indexed {len(docs)} chunks.")
    with col2:
        if st.button("Index File") and uploaded is not None:
            text = uploaded.read().decode("utf-8")
            docs = _split_into_docs(text, title=title or uploaded.name)
            vs = build_faiss_from_docs(docs)
            save_local_faiss(vs, index_dir)
            st.session_state.vs = vs
            st.sidebar.success(f"Indexed {len(docs)} chunks from file.")

    if st.sidebar.button("Load Existing Index"):
        loaded = load_local_faiss(index_dir)
        if loaded is None:
            st.sidebar.error("No index found. Ingest notes first.")
        else:
            st.session_state.vs = loaded
            st.sidebar.success("Index loaded.")


def main():
    st.set_page_config(page_title="Paste and Chat Support Bot", layout="wide")
    ensure_api_key()
    state = get_state()

    index_dir = os.getenv("INDEX_DIR", "index/faiss_index")
    top_k = int(os.getenv("TOP_K", "4"))
    threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.6"))

    sidebar_index_controls(index_dir)

    st.title("Paste and Chat Support Bot")
    st.write("Answers grounded in your notes with citations.")

    if state.vs is None:
        st.info("Index not loaded. Use the sidebar to index or load an index.")
        st.stop()

    retriever = state.vs.as_retriever(search_kwargs={"k": top_k})

    chat_col, src_col = st.columns([2, 1])
    with chat_col:
        user_q = st.text_input("Ask a question", value="How do I reset the connector?")
        if "last_result" not in st.session_state:
            st.session_state.last_result = None
        if st.button("Ask") and user_q.strip():
            st.session_state.last_result = answer_with_guard(
                user_q, retriever=retriever, memory=state.memory, threshold=threshold, top_k=top_k
            )
        if st.session_state.last_result:
            st.markdown(f"**Answer:**\n\n{st.session_state.last_result['answer']}")

    with src_col:
        st.subheader("Source")
        st.caption("Shows the matched chunk and its location.")
        last = st.session_state.last_result
        if last and last.get("quote"):
            st.code(last["quote"], language="markdown")
        if last and last.get("citation"):
            st.text(last["citation"])


if __name__ == "__main__":
    main()
