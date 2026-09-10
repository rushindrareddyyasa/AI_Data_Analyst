from ai_data_analyst.llm.client import generate_response


def generate_rag_answer(
    question: str,
    retrieved_documents: list[dict],
) -> str:
    """
    Generate an answer using only the retrieved documents.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_documents:
        return (
            "I could not find relevant information "
            "in the available documents."
        )

    context_parts = []

    for index, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        context_parts.append(
            f"""
SOURCE {index}: {document['source']}

{document['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are an AI Data Analyst working with company documents.

USER QUESTION
-------------
{question}

RETRIEVED CONTEXT
-----------------
{context}

INSTRUCTIONS
------------

Answer the user's question using ONLY the retrieved context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the retrieved context does not contain enough
   information to answer the question, clearly say so.
4. Give a concise and accurate business answer.
5. Do not mention embeddings, vector databases,
   retrieval, ChromaDB, or internal implementation.
"""

    return generate_response(prompt)