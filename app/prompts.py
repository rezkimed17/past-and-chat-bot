from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def system_prompt() -> str:
    return (
        "You are a support assistant that only answers using the provided notes. "
        "Follow the rules strictly:\n"
        "- Ground every answer in the supplied chunks.\n"
        "- Include exactly one short quoted snippet and a source marker.\n"
        "- If the notes do not contain the answer, say \"I could not find this in your notes\" "
        "and list up to three missing details the user should add.\n"
        "- Be brief, clear, and avoid speculation.\n"
    )


def qa_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt()),
            (
                "human",
                "Notes:\n{context}\n\n"
                "Chat history:\n{history}\n\n"
                "Question: {question}\n\n"
                "Respond concisely. Quote a relevant sentence and add the source marker.",
            ),
        ]
    )

