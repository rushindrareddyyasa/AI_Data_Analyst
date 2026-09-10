from ai_data_analyst.rag.vector_store import VectorStore


def test_vector_store(tmp_path):

    store = VectorStore(
        collection_name="test_documents",
        persist_directory=str(tmp_path / "chroma"),
    )

    chunks = [
        "A region is considered high-performing.",
        "Discounts should be monitored.",
        "Regional performance is evaluated using total sales.",
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    sources = [
        "regional_sales_policy.txt",
        "regional_sales_policy.txt",
        "regional_sales_policy.txt",
    ]

    store.add_documents(
        chunks=chunks,
        embeddings=embeddings,
        sources=sources,
    )

    assert store.count() == 3

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2

    assert results[0]["text"] == (
        "A region is considered high-performing."
    )

    assert results[0]["source"] == (
        "regional_sales_policy.txt"
    )

    assert "distance" in results[0]

def test_vector_store_distance_threshold(tmp_path):

    store = VectorStore(
        collection_name="test_threshold",
        persist_directory=str(tmp_path / "chroma"),
    )

    chunks = [
        "Relevant document.",
        "Another document.",
        "Less relevant document.",
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ]

    sources = [
        "policy.txt",
        "policy.txt",
        "other.txt",
    ]

    store.add_documents(
        chunks=chunks,
        embeddings=embeddings,
        sources=sources,
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=3,
        distance_threshold=0.25,
    )

    assert len(results) == 2

    assert results[0]["text"] == (
        "Relevant document."
    )

    assert results[1]["text"] == (
        "Another document."
    )

    for result in results:
        assert result["distance"] <= 0.25