import pytest

from ai_data_analyst.rag.evaluator import evaluate_retrieval


def test_retrieval_evaluation_success():
    documents = [
        {
            "source": "regional_sales_policy.txt",
            "text": (
                "The regional sales policy defines "
                "sales targets for each region."
            ),
            "distance": 0.1,
        }
    ]

    result = evaluate_retrieval(
        retrieved_documents=documents,
        expected_keywords=[
            "regional",
            "sales",
            "policy",
        ],
        expected_sources=[
            "regional_sales_policy.txt",
        ],
    )

    assert result["documents_retrieved"] == 1
    assert result["retrieval_success"] is True
    assert result["keyword_coverage_score"] == 1.0
    assert result["source_match_score"] == 1.0
    assert result["overall_score"] == 1.0


def test_retrieval_evaluation_no_documents():
    result = evaluate_retrieval(
        retrieved_documents=[],
        expected_keywords=["sales"],
        expected_sources=["policy.txt"],
    )

    assert result["documents_retrieved"] == 0
    assert result["retrieval_success"] is False
    assert result["overall_score"] == 0.0


def test_retrieval_evaluation_partial_keyword_match():
    documents = [
        {
            "source": "policy.txt",
            "text": "The sales policy applies to all regions.",
        }
    ]

    result = evaluate_retrieval(
        retrieved_documents=documents,
        expected_keywords=[
            "sales",
            "policy",
            "employee",
        ],
    )

    assert result["keyword_coverage_score"] == pytest.approx(
        2 / 3
    )