from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """
You are a Retrieval-Augmented Generation (RAG) powered assistant for microwave usage support.
All your answers must use the RAG Context provided with each user message.
The RAG Context will always be presented above a clearly marked "User Question" in the user's message.

Instructions:
- Always rely on the RAG Context content to answer.
- If the user's question is unrelated to microwaves or not covered in the RAG Context, politely refuse to answer and state your scope.
- Do not answer questions that are out of historical conversation scope or outside microwave usage.
- Be factual and concise. Do not invent information beyond the RAG Context.
- Do not mention that you are an AI or refer to the system prompt.
"""

# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """
RAG Context:
{rag_context}

User Question:
{user_question}
"""



DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "vectordb",
    "user": "postgres",
    "password": "postgres",
}

EMBEDDING_MODEL = "text-embedding-3-small-1"
CHAT_MODEL = "gpt-4o"
EMBEDDING_DIMENSIONS = 1536
DEFAULT_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 40
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.5


def run_console_chat() -> None:
    embeddings_client = DialEmbeddingsClient(EMBEDDING_MODEL, API_KEY)
    chat_client = DialChatCompletionClient(CHAT_MODEL, API_KEY)
    text_processor = TextProcessor(embeddings_client, DB_CONFIG)

    text_processor.process_text_file(
        file_path="task/embeddings/microwave_manual.txt",
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_OVERLAP,
        dimensions=EMBEDDING_DIMENSIONS,
        truncate_table=True,
    )

    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    print("Microwave RAG assistant is ready. Type 'exit' to quit.")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            break

        rag_chunks = text_processor.search(
            mode=SearchMode.COSINE_DISTANCE,
            query=user_input,
            top_k=DEFAULT_TOP_K,
            min_score=DEFAULT_MIN_SCORE,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        rag_context = "\n\n---\n\n".join(rag_chunks) if rag_chunks else "No relevant context found."

        user_message_content = USER_PROMPT.format(
            rag_context=rag_context,
            user_question=user_input,
        )

        conversation.add_message(Message(Role.USER, user_message_content))

        ai_message = chat_client.get_completion(conversation.get_messages())
        conversation.add_message(ai_message)

        print(f"Assistant: {ai_message.content}")


run_console_chat()
