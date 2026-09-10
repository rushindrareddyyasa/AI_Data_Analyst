from unittest.mock import patch

from ai_data_analyst.rag.generator import generate_rag_answer


def test_generate_rag_answer():

    question = (
        "When is a region considered high-performing?"
    )

    retrieved_documents = [
        {
            "source": "regional_sales_policy.txt",
            "text": (
                "A region is considered high-performing "
                "when its quarterly sales exceed its "
                "assigned target by more than 10 percent."
            ),
            "distance": 0.19,
        }
    ]

    expected_answer = (
        "A region is considered high-performing when "
        "its quarterly sales exceed its assigned target "
        "by more than 10 percent."
    )

    with patch(
        "ai_data_analyst.rag.generator.generate_response"
    ) as mock_generate:

        mock_generate.return_value = expected_answer

        answer = generate_rag_answer(
            question=question,
            retrieved_documents=retrieved_documents,
        )

        assert answer == expected_answer

        mock_generate.assert_called_once()