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


def _condense_question(memory, question: str) -> str:
    """Rewrite follow-ups into standalone questions using chat history."""
    try:
        vars = memory.load_memory_variables({})
        msgs = vars.get("chat_history", [])
        history_lines: List[str] = []
        for m in msgs:
            name = getattr(m, "type", "") or m.__class__.__name__
            role = "User" if "Human" in name else ("Assistant" if "AI" in name else "Message")
            content = getattr(m, "content", str(m))
            history_lines.append(f"{role}: {content}")
        history = "\n".join(history_lines[-8:])  # last few turns
        prompt = (
            "Rewrite the user's question to be a self-contained query given this chat history.\n"
            f"Chat History:\n{history}\n\nQuestion: {question}\n\nStandalone:"
        )
        llm = _llm()
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", "").strip()
        return text or question
    except Exception:
        return question

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
    effective_q = _condense_question(memory, question)
    docs_scores: List[Tuple[Document, float]] = retriever.vectorstore.similarity_search_with_score(
        query=effective_q, k=top_k
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
