from unittest.mock import MagicMock

from ai_data_analyst.rag.embeddings import EmbeddingModel


def test_embed_document():

    mock_client = MagicMock()

    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]

    mock_result = MagicMock()
    mock_result.embeddings = [mock_embedding]

    mock_client.models.embed_content.return_value = mock_result

    embedding_model = EmbeddingModel(mock_client)

    result = embedding_model.embed_document(
        "A region is considered high-performing."
    )

    assert result == [0.1, 0.2, 0.3]

    mock_client.models.embed_content.assert_called_once()


def test_embed_query():

    mock_client = MagicMock()

    mock_embedding = MagicMock()
    mock_embedding.values = [0.4, 0.5, 0.6]

    mock_result = MagicMock()
    mock_result.embeddings = [mock_embedding]

    mock_client.models.embed_content.return_value = mock_result

    embedding_model = EmbeddingModel(mock_client)

    result = embedding_model.embed_query(
        "What makes a region high-performing?"
    )

    assert result == [0.4, 0.5, 0.6]

    mock_client.models.embed_content.assert_called_once()