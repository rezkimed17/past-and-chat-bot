from __future__ import annotations

from langchain_core.documents import Document

from app.memory import build_memory
from app.chains import answer_with_guard
from app.vectorstore import build_faiss_from_docs


def test_refuses_when_no_support(monkeypatch):
    docs = [Document(page_content="Only info about connectors.", metadata={"title": "Guide"})]
    vs = build_faiss_from_docs(docs)
    retriever = vs.as_retriever(search_kwargs={"k": 2})

    # Use a very high threshold to force refusal regardless of content
    result = answer_with_guard(
        question="What are warranty terms?",
        retriever=retriever,
        memory=build_memory(),
        threshold=0.99,
        top_k=2,
    )
    assert "I could not find this in your notes" in result["answer"]

