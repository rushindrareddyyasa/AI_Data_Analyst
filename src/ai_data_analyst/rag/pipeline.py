from ai_data_analyst.rag.retriever import Retriever
from ai_data_analyst.rag.generator import generate_rag_answer


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Pipeline:

        User Question
              ↓
        Retriever
              ↓
        Relevant Chunks
              ↓
        RAG Generator
              ↓
        Grounded Answer
    """

    def __init__(
        self,
        retriever: Retriever,
    ):
        """
        Initialize the RAG pipeline.
        """

        if retriever is None:
            raise ValueError(
                "Retriever cannot be None."
            )

        self.retriever = retriever


    # ==================================================
    # VALIDATE QUESTION
    # ==================================================

    @staticmethod
    def _validate_question(
        question: str,
    ) -> None:
        """
        Validate the user's question.
        """

        if not question or not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )


    # ==================================================
    # RETRIEVE DOCUMENTS
    # ==================================================

    def _retrieve_documents(
        self,
        question: str,
        top_k: int,
    ) -> list[dict]:
        """
        Retrieve relevant document chunks.
        """

        if not isinstance(
            top_k,
            int,
        ):

            raise TypeError(
                "top_k must be an integer."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0."
            )

        return self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )


    # ==================================================
    # GENERATE ANSWER
    # ==================================================

    @staticmethod
    def _generate_answer(
        question: str,
        retrieved_documents: list[dict],
    ) -> str:
        """
        Generate an answer using the retrieved context.

        The generator is responsible for deciding how to
        handle cases where no relevant documents were
        retrieved.
        """

        return generate_rag_answer(
            question=question,
            retrieved_documents=retrieved_documents,
        )


    # ==================================================
    # EXTRACT SOURCES
    # ==================================================

    @staticmethod
    def _extract_sources(
        retrieved_documents: list[dict],
    ) -> list[str]:
        """
        Extract unique source filenames while preserving
        retrieval order.
        """

        sources = []

        for document in retrieved_documents:

            source = document.get(
                "source"
            )

            if (
                source
                and source not in sources
            ):

                sources.append(
                    source
                )

        return sources


    # ==================================================
    # BUILD RETRIEVAL EVIDENCE
    # ==================================================

    @staticmethod
    def _build_retrieval_evidence(
        retrieved_documents: list[dict],
    ) -> list[dict]:
        """
        Build a clean representation of retrieved
        evidence.

        Keeps the existing document text while exposing:

        - rank
        - source
        - distance
        - relevance score
        """

        evidence = []

        for index, document in enumerate(
            retrieved_documents,
            start=1,
        ):

            evidence.append(
                {
                    "rank": index,

                    "source": document.get(
                        "source"
                    ),

                    "text": document.get(
                        "text",
                        "",
                    ),

                    "distance": document.get(
                        "distance"
                    ),

                    "relevance_score": document.get(
                        "relevance_score"
                    ),
                }
            )

        return evidence


    # ==================================================
    # ANSWER
    # ==================================================

    def answer(
        self,
        question: str,
        top_k: int = 3,
    ) -> str:
        """
        Retrieve relevant documents and generate
        an answer from them.

        Returns only the final answer.
        """

        self._validate_question(
            question
        )

        retrieved_documents = (
            self._retrieve_documents(
                question=question,
                top_k=top_k,
            )
        )

        # --------------------------------------------------
        # Always pass retrieved documents to the generator.
        #
        # This includes an empty list. The generator is
        # responsible for handling the no-context case.
        # --------------------------------------------------

        return self._generate_answer(
            question=question,
            retrieved_documents=retrieved_documents,
        )


    # ==================================================
    # ANSWER WITH SOURCES
    # ==================================================

    def answer_with_sources(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict:
        """
        Retrieve relevant documents and generate an
        answer together with source information.

        Returns:

        {
            "answer": "...",
            "sources": [...],
            "retrieved_documents": [...]
        }
        """

        self._validate_question(
            question
        )

        retrieved_documents = (
            self._retrieve_documents(
                question=question,
                top_k=top_k,
            )
        )

        # --------------------------------------------------
        # Generate answer
        # --------------------------------------------------

        answer = self._generate_answer(
            question=question,
            retrieved_documents=retrieved_documents,
        )

        # --------------------------------------------------
        # Extract sources
        # --------------------------------------------------

        sources = self._extract_sources(
            retrieved_documents
        )

        # --------------------------------------------------
        # Build retrieval evidence
        # --------------------------------------------------

        retrieval_evidence = (
            self._build_retrieval_evidence(
                retrieved_documents
            )
        )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_documents": retrieval_evidence,
        }