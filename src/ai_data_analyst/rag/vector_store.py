import chromadb


class VectorStore:
    """
    Persistent ChromaDB vector store.
    """

    def __init__(
        self,
        collection_name: str = "business_documents",
        persist_directory: str = "data/processed/chroma_db",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name
            )
        )

    def add_documents(
        self,
        chunks,
        embeddings,
        sources,
    ):
        """
        Add document chunks and their embeddings
        to the vector store.
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

        ids = [
            f"chunk_{index}"
            for index in range(len(chunks))
        ]

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
        Return the number of stored documents.
        """

        return self.collection.count()

    def search(
        self,
        query_embedding,
        top_k: int = 3,
        distance_threshold: float | None = None,
    ) -> list[dict]:
        """
        Search the vector store.

        If distance_threshold is provided, results with
        a distance greater than the threshold are removed.

        Lower Chroma distance means greater similarity.
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
                    "source": metadata.get(
                        "source"
                    ),
                    "distance": distance,
                }
            )

        return retrieved_documents

    def reset(self):
        """
        Delete the current collection and recreate it.

        Useful when rebuilding the index after changing
        chunking or embedding configuration.
        """

        collection_name = (
            self.collection.name
        )

        self.client.delete_collection(
            name=collection_name
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name
            )
        )