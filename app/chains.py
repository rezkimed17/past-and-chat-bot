from __future__ import annotations

import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from .prompts import qa_prompt
from .utils import best_supported, format_citation, quote_snippet, suggest_missing_details


def _llm() -> BaseChatModel:
    load_dotenv()
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("TEMPERATURE", "0"))
    return ChatOpenAI(model=model, temperature=temperature)


def build_conv_chain(retriever, memory) -> ConversationalRetrievalChain:
    chain = ConversationalRetrievalChain.from_llm(
        llm=_llm(),
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt()},
        return_source_documents=True,
        verbose=False,
    )
    return chain


def answer_with_guard(
    question: str,
    retriever,
    memory,
    threshold: float,
    top_k: int = 4,
) -> Dict[str, str]:
    """Retrieve, apply threshold guard, then answer with citation or refuse.

    Returns dict with keys: answer, citation, quote.
    """
    # Pull scored docs directly for guard
    docs_scores: List[Tuple[Document, float]] = retriever.vectorstore.similarity_search_with_relevance_scores(
        query=question, k=top_k
    )
    supported = best_supported(docs_scores, threshold=threshold)

    if not supported:
        miss = suggest_missing_details(question)
        refusal = "I could not find this in your notes.\nMissing details to add: " + miss
        return {"answer": refusal, "citation": "", "quote": ""}

    # Use the standard chain for grounded answer
    chain = build_conv_chain(retriever=retriever, memory=memory)
    result = chain.invoke({"question": question})

    # Prefer the highest-scoring supported document for citation/quote
    top_doc = sorted(supported, key=lambda s: s.score, reverse=True)[0].doc
    citation = format_citation(top_doc)
    quote = quote_snippet(top_doc)
    answer = result.get("answer", "").strip()
    if citation and quote:
        answer = f"{answer}\n{quote} {citation}"
    return {"answer": answer, "citation": citation, "quote": quote}
