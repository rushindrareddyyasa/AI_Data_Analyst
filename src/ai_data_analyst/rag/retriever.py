from typing import Optional


class Retriever:
    """
    Retrieve relevant document chunks from the vector store.

    Responsibilities
    ----------------
    1. Validate the user's query.
    2. Generate a query embedding.
    3. Perform semantic search using the vector store.
    4. Apply the configured distance threshold.
    5. Enrich results with a relevance score.

    Expected vector-store result format:

    {
        "text": "...",
        "source": "...",
        "distance": 0.15
    }
    """

    def __init__(
        self,
        embedding_model,
        vector_store,
        distance_threshold: Optional[float] = None,
    ):
        """
        Initialize the retriever.

        Parameters
        ----------
        embedding_model:
            Object responsible for generating query embeddings.

        vector_store:
            Vector database used for semantic search.

        distance_threshold:
            Optional maximum distance allowed for retrieved
            chunks. Smaller distance generally indicates
            greater semantic similarity.
        """

        if embedding_model is None:
            raise ValueError(
                "Embedding model cannot be None."
            )

        if vector_store is None:
            raise ValueError(
                "Vector store cannot be None."
            )

        if (
            distance_threshold is not None
            and distance_threshold < 0
        ):
            raise ValueError(
                "distance_threshold cannot be negative."
            )

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.distance_threshold = distance_threshold


    # ==================================================
    # RETRIEVE
    # ==================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Retrieve the most relevant document chunks.

        Parameters
        ----------
        query:
            User's natural-language question.

        top_k:
            Maximum number of chunks to retrieve.

        Returns
        -------
        list[dict]

            Each result contains:

            - text
            - source
            - distance
            - relevance_score
        """

        # --------------------------------------------------
        # Validate query
        # --------------------------------------------------

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )


        # --------------------------------------------------
        # Validate top_k
        # --------------------------------------------------

        if not isinstance(top_k, int):

            raise TypeError(
                "top_k must be an integer."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0."
            )


        # --------------------------------------------------
        # Check whether vector store contains documents
        # --------------------------------------------------

        if self.vector_store.count() == 0:

            return []


        # --------------------------------------------------
        # Generate query embedding
        # --------------------------------------------------

        query_embedding = (
            self.embedding_model.embed_query(
                query.strip()
            )
        )


        # --------------------------------------------------
        # Semantic search
        # --------------------------------------------------

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            distance_threshold=(
                self.distance_threshold
            ),
        )


        # --------------------------------------------------
        # Enrich retrieval results
        # --------------------------------------------------

        enriched_results = []

        for result in results:

            distance = result.get(
                "distance"
            )

            # ----------------------------------------------
            # Calculate normalized relevance score
            # ----------------------------------------------

            if distance is None:

                relevance_score = None

            else:

                relevance_score = (
                    1
                    / (
                        1
                        + max(
                            float(distance),
                            0.0,
                        )
                    )
                )


            # ----------------------------------------------
            # Preserve existing result contract
            # ----------------------------------------------

            enriched_result = {
                "text": result.get(
                    "text",
                    "",
                ),

                "source": result.get(
                    "source",
                    "Unknown",
                ),

                "distance": distance,

                "relevance_score": relevance_score,
            }

            enriched_results.append(
                enriched_result
            )


        return enriched_results