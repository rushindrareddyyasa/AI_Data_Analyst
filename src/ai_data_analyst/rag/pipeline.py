from ai_data_analyst.rag.retriever import Retriever
from ai_data_analyst.rag.generator import generate_rag_answer


class RAGPipeline:

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def answer(
        self,
        question: str,
        top_k: int = 3,
    ) -> str:
        """
        Retrieve relevant documents and generate
        an answer from them.
        """

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        retrieved_documents = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        answer = generate_rag_answer(
            question=question,
            retrieved_documents=retrieved_documents,
        )

        return answer

    def answer_with_sources(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict:
        """
        Retrieve relevant documents and generate an answer
        together with the source documents.
        """

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        retrieved_documents = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        answer = generate_rag_answer(
            question=question,
            retrieved_documents=retrieved_documents,
        )

        sources = []

        for document in retrieved_documents:
            source = document.get("source")

            if source and source not in sources:
                sources.append(source)

        return {
            "answer": answer,
            "sources": sources,
        }