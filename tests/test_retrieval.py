from __future__ import annotations

from langchain_core.documents import Document

from app.vectorstore import build_faiss_from_docs


def test_retrieval_matches_relevant_text():
    docs = [
        Document(page_content="Reset the connector by unplugging for 10 seconds.", metadata={"title": "Guide"}),
        Document(page_content="Unrelated info about colors.", metadata={"title": "Misc"}),
    ]
    vs = build_faiss_from_docs(docs)
    results = vs.similarity_search_with_relevance_scores("How to reset the connector?", k=2)
    assert results[0][0].page_content.startswith("Reset the connector")
    assert results[0][1] >= results[1][1]

