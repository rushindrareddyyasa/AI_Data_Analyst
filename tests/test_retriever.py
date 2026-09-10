from unittest.mock import MagicMock

from ai_data_analyst.rag.retriever import Retriever


def test_retrieve():

    embedding_model = MagicMock()
    vector_store = MagicMock()

    embedding_model.embed_query.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    vector_store.search.return_value = [
        {
            "text": "A region is considered high-performing.",
            "source": "regional_sales_policy.txt",
            "distance": 0.1,
        },
        {
            "text": "Regional performance is evaluated using total sales.",
            "source": "regional_sales_policy.txt",
            "distance": 0.2,
        },
    ]

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "What makes a region high-performing?",
        top_k=2,
    )

    assert len(results) == 2

    assert results[0]["text"] == (
        "A region is considered high-performing."
    )

    embedding_model.embed_query.assert_called_once_with(
        "What makes a region high-performing?"
    )

    vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=2,
        distance_threshold=None,
    )

def test_retrieve_with_distance_threshold():
    embedding_model = MagicMock()
    vector_store = MagicMock()

    embedding_model.embed_query.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    vector_store.search.return_value = [
        {
            "text": "Relevant document.",
            "source": "policy.txt",
            "distance": 0.1,
        }
    ]

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        distance_threshold=0.25,
    )

    results = retriever.retrieve(
        "What is the policy?",
        top_k=3,
    )

    assert len(results) == 1

    vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=3,
        distance_threshold=0.25,
    )