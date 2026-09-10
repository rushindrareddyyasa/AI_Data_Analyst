from unittest.mock import MagicMock, patch

from ai_data_analyst.rag.pipeline import RAGPipeline


def test_rag_pipeline():

    retriever = MagicMock()

    retriever.retrieve.return_value = [
        {
            "text": (
                "A region is considered high-performing "
                "when its quarterly sales exceed its "
                "assigned target by more than 10 percent."
            ),
            "source": "regional_sales_policy.txt",
            "distance": 0.19,
        }
    ]

    expected_answer = (
        "A region is considered high-performing when "
        "its quarterly sales exceed its assigned target "
        "by more than 10 percent."
    )

    with patch(
        "ai_data_analyst.rag.pipeline.generate_rag_answer"
    ) as mock_generate:

        mock_generate.return_value = expected_answer

        pipeline = RAGPipeline(
            retriever=retriever
        )

        answer = pipeline.answer(
            "When is a region considered high-performing?",
            top_k=2,
        )

        assert answer == expected_answer

        retriever.retrieve.assert_called_once_with(
            query=(
                "When is a region considered high-performing?"
            ),
            top_k=2,
        )

        mock_generate.assert_called_once()

def test_rag_pipeline_no_relevant_documents():

    retriever = MagicMock()

    retriever.retrieve.return_value = []

    pipeline = RAGPipeline(
        retriever=retriever
    )

    with patch(
        "ai_data_analyst.rag.pipeline.generate_rag_answer"
    ) as mock_generate:

        mock_generate.return_value = (
            "I could not find relevant information "
            "in the available documents."
        )

        answer = pipeline.answer(
            "What is the employee leave policy?",
            top_k=3,
        )

        assert answer == (
            "I could not find relevant information "
            "in the available documents."
        )

        retriever.retrieve.assert_called_once_with(
            query="What is the employee leave policy?",
            top_k=3,
        )

        mock_generate.assert_called_once_with(
            question="What is the employee leave policy?",
            retrieved_documents=[],
        )