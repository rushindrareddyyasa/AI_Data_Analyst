from ai_data_analyst.llm.client import client
from ai_data_analyst.rag.embeddings import EmbeddingModel
from ai_data_analyst.rag.vector_store import VectorStore
from ai_data_analyst.rag.retriever import Retriever


def test_real_retrieval():

    embedding_model = EmbeddingModel(client)

    vector_store = VectorStore()

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        "When is a region considered high-performing?",
        top_k=2,
    )

    assert len(results) > 0

    print("\n========== RETRIEVED RESULTS ==========\n")

    for result in results:
        print("SOURCE:", result["source"])
        print("DISTANCE:", result["distance"])
        print("TEXT:")
        print(result["text"])
        print("-" * 60)

    assert any(
        "high-performing" in result["text"].lower()
        for result in results
    )