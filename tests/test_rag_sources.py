from unittest.mock import MagicMock, patch

from ai_data_analyst.rag.pipeline import RAGPipeline


def test_rag_pipeline_sources():

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
        },
        {
            "text": (
                "Regional performance is evaluated using "
                "total sales generated during the relevant quarter."
            ),
            "source": "regional_sales_policy.txt",
            "distance": 0.25,
        },
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

        result = pipeline.answer_with_sources(
            "When is a region considered high-performing?",
            top_k=2,
        )

        assert result["answer"] == expected_answer

        assert result["sources"] == [
            "regional_sales_policy.txt"
        ]