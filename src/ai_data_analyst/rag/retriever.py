from typing import Optional


class Retriever:
    """
    Retrieve relevant documents from the vector store.
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
        distance_threshold: Optional[float] = None,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.distance_threshold = distance_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Retrieve relevant documents for a query.

        Results can optionally be filtered using the
        configured distance threshold.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_embedding = (
            self.embedding_model.embed_query(
                query
            )
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            distance_threshold=(
                self.distance_threshold
            ),
        )

        return results