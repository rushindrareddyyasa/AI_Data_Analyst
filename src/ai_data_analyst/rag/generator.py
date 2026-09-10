from ai_data_analyst.llm.client import generate_response


def generate_rag_answer(
    question: str,
    retrieved_documents: list[dict],
) -> str:
    """
    Generate a grounded answer using retrieved documents.

    The LLM is instructed to use only the provided context
    and avoid unsupported claims.
    """

    # ---------------------------------------------------------
    # 1. Validate question
    # ---------------------------------------------------------
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    # ---------------------------------------------------------
    # 2. Handle no retrieved documents
    # ---------------------------------------------------------
    if not retrieved_documents:
        return (
            "I could not find relevant information "
            "in the available documents."
        )

    # ---------------------------------------------------------
    # 3. Build context from retrieved documents
    # ---------------------------------------------------------
    context_parts = []

    for index, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        if not isinstance(document, dict):
            continue

        source = document.get("source", "Unknown source")
        text = document.get("text", "").strip()

        if not text:
            continue

        context_parts.append(
            f"""
SOURCE {index}
------------
Source: {source}

{text}
"""
        )

    # All retrieved documents were empty/invalid.
    if not context_parts:
        return (
            "I could not find relevant information "
            "in the available documents."
        )

    context = "\n".join(context_parts)

    # ---------------------------------------------------------
    # 4. Build grounded RAG prompt
    # ---------------------------------------------------------
    prompt = f"""
You are an AI Data Analyst answering questions using
company-provided documents.

Your task is to answer the user's question using ONLY
the information contained in the provided context.

USER QUESTION
-------------
{question.strip()}

RETRIEVED CONTEXT
-----------------
{context}

ANSWERING RULES
---------------

1. Use only the retrieved context to answer the question.

2. Do not use outside knowledge, assumptions, or guesses.

3. Do not invent facts, numbers, policies, dates, names,
   or other information that is not present in the context.

4. If the context does not contain enough information,
   clearly state:
   "The available documents do not contain enough
   information to answer this question."

5. If only part of the question can be answered,
   answer the supported part and clearly identify
   what information is missing.

6. Keep the answer concise, clear, and business-friendly.

7. When useful, mention the source document that supports
   the answer.

8. Do not mention internal implementation details such as:
   - embeddings
   - vector databases
   - ChromaDB
   - retrieval algorithms
   - similarity scores
   - RAG pipelines
   - internal system architecture

9. Never claim that information exists in the documents
   unless it is actually present in the retrieved context.

FINAL ANSWER
------------
"""

    # ---------------------------------------------------------
    # 5. Generate answer
    # ---------------------------------------------------------
    answer = generate_response(prompt)

    # ---------------------------------------------------------
    # 6. Validate generated response
    # ---------------------------------------------------------
    if not answer or not answer.strip():
        return (
            "I could not generate an answer from "
            "the available documents."
        )

    return answer.strip()