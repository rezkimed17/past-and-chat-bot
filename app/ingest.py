from __future__ import annotations

import argparse
import os
import sys
from typing import List

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .utils import clean_text, compute_line_spans
from .vectorstore import build_faiss_from_docs, load_local_faiss, save_local_faiss


def _split_into_docs(text: str, title: str) -> List[Document]:
    text = clean_text(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_text(text)
    spans = compute_line_spans(text, chunks)
    docs: List[Document] = []
    for i, (chunk, (start, end)) in enumerate(zip(chunks, spans)):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "title": title,
                    "chunk_index": i,
                    "start_line": start,
                    "end_line": end,
                },
            )
        )
    return docs


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest notes into FAISS index.")
    parser.add_argument("--file", type=str, help="Path to local file to index", default=None)
    parser.add_argument("--stdin", action="store_true", help="Read text from STDIN")
    parser.add_argument("--title", type=str, required=True, help="Title/source label for the notes")
    parser.add_argument(
        "--index-dir",
        type=str,
        default=os.getenv("INDEX_DIR", "index/faiss_index"),
        help="Directory to save the FAISS index",
    )
    args = parser.parse_args()

    if not args.file and not args.stdin:
        parser.error("Specify --file or --stdin")

    raw_text = ""
    if args.file:
        if not os.path.exists(args.file):
            sys.exit(f"File not found: {args.file}")
        with open(args.file, "r", encoding="utf-8") as f:
            raw_text = f.read()
    elif args.stdin:
        raw_text = sys.stdin.read()

    if not raw_text.strip():
        sys.exit("No input text provided.")

    docs = _split_into_docs(raw_text, title=args.title)

    existing = load_local_faiss(args.index_dir)
    if existing is None:
        vs = build_faiss_from_docs(docs)
        save_local_faiss(vs, args.index_dir)
        print(f"Created index with {len(docs)} chunks at {args.index_dir}")
    else:
        existing.add_documents(docs)
        save_local_faiss(existing, args.index_dir)
        print(f"Updated index with {len(docs)} new chunks at {args.index_dir}")


if __name__ == "__main__":
    main()

