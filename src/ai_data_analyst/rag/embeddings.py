from google import genai
from google.genai import types


EMBEDDING_MODEL = "gemini-embedding-001"


class EmbeddingModel:

    def __init__(self, client):
        self.client = client

    def embed_document(self, text: str) -> list[float]:
        """
        Generate an embedding for a document chunk.
        """

        result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )

        return result.embeddings[0].values

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a user search query.
        """

        result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )

        return result.embeddings[0].values