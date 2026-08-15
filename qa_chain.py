from langchain_core.documents import Document
from langchain_groq import ChatGroq

FALLBACK_RESPONSE = "Not found in the document."


def get_llm(api_key: str, model_name: str = "llama-3.1-8b-instant") -> ChatGroq:
    """
    Instantiate a ChatGroq LLM.
    api_key is passed explicitly — never read from env inside this function.
    """
    return ChatGroq(api_key=api_key, model_name=model_name)


def build_prompt_text(context_chunks: list[Document], question: str) -> str:
    """
    Construct the complete prompt string to send to the LLM.

    The prompt contains:
      1. The grounding instruction: "Answer using ONLY the document context below."
      2. The fallback instruction including the verbatim fallback string.
      3. All chunk texts from context_chunks, separated by "\\n\\n---\\n\\n".
      4. The user's question.
    """
    chunk_texts = "\n\n---\n\n".join(chunk.page_content for chunk in context_chunks)

    return (
        "You are a document assistant. Your task is to answer the user's question\n"
        "using ONLY the document context provided below.\n"
        "\n"
        "Rules:\n"
        "- Answer using only the information in the context.\n"
        "- Do not use any external knowledge or make up information.\n"
        "- If the answer is not present in the context, respond with exactly this\n"
        f"  string and nothing else: {FALLBACK_RESPONSE}\n"
        "\n"
        "Context:\n"
        f"{chunk_texts}\n"
        "\n"
        f"Question: {question}\n"
        "\n"
        "Answer:"
    )


def run_qa_chain(
    question: str,
    retriever,
    llm: ChatGroq,
) -> tuple[str, list[Document]]:
    """
    Execute a full QA query:
      1. Retrieve chunks from the retriever.
      2. If no chunks returned, return (FALLBACK_RESPONSE, []) without calling LLM.
      3. Build prompt from chunks and question.
      4. Call LLM exactly once; raise RuntimeError on failure.
      5. Return (answer_string, source_docs).
    """
    chunks: list[Document] = retriever.invoke(question)

    if not chunks:
        return (FALLBACK_RESPONSE, [])

    prompt_text = build_prompt_text(chunks, question)

    try:
        response = llm.invoke(prompt_text)
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    answer_str: str = response.content
    return (answer_str, chunks)
