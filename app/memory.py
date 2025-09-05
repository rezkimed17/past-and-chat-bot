from __future__ import annotations

from langchain.memory import ConversationBufferMemory


def build_memory() -> ConversationBufferMemory:
    """Conversation buffer memory configured for ConversationalRetrievalChain.

    - memory_key: where chat history is stored
    - input_key: name of the user input field
    - output_key: name of the LLM answer field
    - return_messages: provide messages for internal chains
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )
