from __future__ import annotations

from langchain.memory import ConversationBufferMemory


def build_memory() -> ConversationBufferMemory:
    """Conversation buffer memory storing plain text history."""
    return ConversationBufferMemory(memory_key="history", return_messages=False)

