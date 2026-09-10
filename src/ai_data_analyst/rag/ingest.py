from ai_data_analyst.rag.loader import (
    load_documents,
    load_uploaded_document,
)
from ai_data_analyst.rag.chunker import chunk_document
from ai_data_analyst.rag.embeddings import EmbeddingModel
from ai_data_analyst.rag.vector_store import VectorStore
from ai_data_analyst.llm.client import client


def _ingest_document_list(
    documents: list[dict],
    vector_store: VectorStore | None = None,
) -> VectorStore:
    """
    Process documents through the complete RAG ingestion pipeline.

    Pipeline:
        Documents
            ↓
        Chunking
            ↓
        Embeddings
            ↓
        ChromaDB

    Parameters
    ----------
    documents:
        List of documents containing:
        - source
        - text

    vector_store:
        Existing VectorStore instance.
        If None, a new VectorStore is created.

    Returns
    -------
    VectorStore
        The vector store containing the ingested chunks.
    """

    if not documents:
        raise ValueError("No documents were provided for ingestion.")

    if vector_store is None:
        vector_store = VectorStore()

    embedding_model = EmbeddingModel(client)

    all_chunks = []
    all_embeddings = []
    all_sources = []

    for document in documents:
        source = document.get("source")
        text = document.get("text")

        if not source:
            raise ValueError("Document source is missing.")

        if not text or not text.strip():
            continue

        chunks = chunk_document(text)

        for chunk in chunks:
            embedding = embedding_model.embed_document(chunk)

            all_chunks.append(chunk)
            all_embeddings.append(embedding)
            all_sources.append(source)

    if all_chunks:
        vector_store.add_documents(
            chunks=all_chunks,
            embeddings=all_embeddings,
            sources=all_sources,
        )

    return vector_store


def ingest_documents(
    directory: str = "data/raw/documents",
) -> VectorStore:
    """
    Load documents from a directory and ingest them
    into the ChromaDB vector store.
    """

    documents = load_documents(directory)

    return _ingest_document_list(
        documents=documents,
    )


def ingest_uploaded_documents(
    uploaded_files,
    vector_store: VectorStore | None = None,
) -> VectorStore:
    """
    Ingest documents uploaded through Streamlit.

    Parameters
    ----------
    uploaded_files:
        A single uploaded file or a collection of
        uploaded files.

    vector_store:
        Optional existing VectorStore instance.

    Returns
    -------
    VectorStore
        Vector store containing the uploaded documents.
    """

    if uploaded_files is None:
        raise ValueError("No uploaded documents were provided.")

    if not isinstance(uploaded_files, (list, tuple)):
        uploaded_files = [uploaded_files]

    documents = []

    for uploaded_file in uploaded_files:
        document = load_uploaded_document(
            uploaded_file
        )

        documents.append(document)

    return _ingest_document_list(
        documents=documents,
        vector_store=vector_store,
    )