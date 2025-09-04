from __future__ import annotations

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


def _embeddings() -> OpenAIEmbeddings:
    load_dotenv()
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def save_local_faiss(vs: FAISS, index_dir: str) -> None:
    os.makedirs(index_dir, exist_ok=True)
    vs.save_local(index_dir)


def load_local_faiss(index_dir: str) -> Optional[FAISS]:
    if not os.path.isdir(index_dir):
        return None
    try:
        return FAISS.load_local(index_dir, _embeddings(), allow_dangerous_deserialization=True)
    except Exception:
        return None


def build_faiss_from_docs(docs: List[Document]) -> FAISS:
    return FAISS.from_documents(documents=docs, embedding=_embeddings())

