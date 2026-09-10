import hashlib

import chromadb


class VectorStore:
    """
    Persistent ChromaDB vector store.

    Stores document chunks, embeddings, and source metadata
    for semantic retrieval.
    """

    def __init__(
        self,
        collection_name: str = "business_documents",
        persist_directory: str = "data/processed/chroma_db",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        sources: list[str],
    ) -> None:
        """
        Add document chunks and their embeddings to ChromaDB.

        Each chunk receives a deterministic ID based on:
        - source filename
        - chunk index
        - chunk content

        This prevents ID collisions when multiple documents
        are ingested into the same collection.
        """

        if not (
            len(chunks)
            == len(embeddings)
            == len(sources)
        ):
            raise ValueError(
                "chunks, embeddings, and sources "
                "must have the same length."
            )

        if not chunks:
            return

        ids = []

        for index, (chunk, source) in enumerate(
            zip(chunks, sources)
        ):
            unique_text = (
                f"{source}__{index}__{chunk}"
            )

            chunk_hash = hashlib.sha256(
                unique_text.encode("utf-8")
            ).hexdigest()[:16]

            chunk_id = (
                f"{source}__chunk_{index}__{chunk_hash}"
            )

            ids.append(chunk_id)

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": source
                }
                for source in sources
            ],
        )

    def count(self) -> int:
        """
        Return the number of stored document chunks.
        """

        return self.collection.count()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        distance_threshold: float | None = None,
    ) -> list[dict]:
        """
        Search the vector store using semantic similarity.

        Parameters
        ----------
        query_embedding:
            Embedding vector representing the user's query.

        top_k:
            Maximum number of chunks to retrieve.

        distance_threshold:
            Optional maximum ChromaDB distance.
            Lower distance means greater similarity.

        Returns
        -------
        list[dict]
            Retrieved documents containing:
            - text
            - source
            - distance
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if (
            distance_threshold is not None
            and distance_threshold < 0
        ):
            raise ValueError(
                "distance_threshold cannot be negative."
            )

        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        retrieved_documents = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            if (
                distance_threshold is not None
                and distance > distance_threshold
            ):
                continue

            retrieved_documents.append(
                {
                    "text": document,
                    "source": metadata.get("source"),
                    "distance": distance,
                }
            )

        return retrieved_documents

    def reset(self) -> None:
        """
        Delete the current collection and recreate it.

        Useful when rebuilding the complete vector index.
        """

        collection_name = self.collection.name

        self.client.delete_collection(
            name=collection_name
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )