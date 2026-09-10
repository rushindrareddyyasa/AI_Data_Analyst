from ai_data_analyst.rag.loader import load_documents
from ai_data_analyst.rag.chunker import chunk_document
from ai_data_analyst.rag.embeddings import EmbeddingModel
from ai_data_analyst.rag.vector_store import VectorStore
from ai_data_analyst.llm.client import client


def ingest_documents(
    directory: str = "data/raw/documents",
):
    """
    Load documents, split them into chunks, generate embeddings,
    and store the chunks in ChromaDB.
    """

    documents = load_documents(directory)

    embedding_model = EmbeddingModel(client)

    vector_store = VectorStore()

    all_chunks = []
    all_embeddings = []
    all_sources = []

    for document in documents:

        chunks = chunk_document(
            document["text"]
        )

        for chunk in chunks:

            embedding = embedding_model.embed_document(
                chunk
            )

            all_chunks.append(chunk)
            all_embeddings.append(embedding)
            all_sources.append(document["source"])

    if all_chunks:

        vector_store.add_documents(
            chunks=all_chunks,
            embeddings=all_embeddings,
            sources=all_sources,
        )

    return vector_store