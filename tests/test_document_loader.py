from ai_data_analyst.rag.loader import load_documents


def test_load_documents():

    documents = load_documents(
        "data/raw/documents"
    )

    assert len(documents) == 1

    assert documents[0]["source"] == (
        "regional_sales_policy.txt"
    )

    assert "REGIONAL SALES PERFORMANCE POLICY" in (
        documents[0]["text"]
    )