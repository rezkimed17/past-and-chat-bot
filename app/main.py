from __future__ import annotations

import os
import sys
from typing import Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from .memory import build_memory
from .chains import answer_with_guard
from .vectorstore import load_local_faiss


def _env_or_die() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Create .env and set the key.")


def _load_vs_or_die(index_dir: str) -> FAISS:
    vs = load_local_faiss(index_dir)
    if vs is None:
        sys.exit("Index not found. Run app/ingest.py to build it.")
    return vs


def main() -> None:
    _env_or_die()
    index_dir = os.getenv("INDEX_DIR", "index/faiss_index")
    top_k = int(os.getenv("TOP_K", "4"))
    threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))

    vs = _load_vs_or_die(index_dir)
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    memory = build_memory()

    print("Paste and Chat Support Bot (CLI)")
    print("Type your question. Ctrl+C or 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if q.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not q:
            continue
        result = answer_with_guard(q, retriever=retriever, memory=memory, threshold=threshold, top_k=top_k)
        print(f"Bot: {result['answer']}\n")


if __name__ == "__main__":
    main()
