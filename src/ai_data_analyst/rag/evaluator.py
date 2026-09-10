from typing import Optional


def _validate_documents(
    retrieved_documents: list[dict],
) -> None:
    """Validate the retrieved document structure."""

    if not isinstance(retrieved_documents, list):
        raise ValueError("Retrieved documents must be a list.")

    for document in retrieved_documents:
        if not isinstance(document, dict):
            raise ValueError(
                "Each retrieved document must be a dictionary."
            )


def _normalize_text(text: str) -> str:
    """Normalize text for simple keyword comparison."""

    return " ".join(
        text.lower().strip().split()
    )


def evaluate_retrieval(
    retrieved_documents: list[dict],
    expected_keywords: Optional[list[str]] = None,
    expected_sources: Optional[list[str]] = None,
) -> dict:
    """
    Evaluate the quality of retrieved RAG documents.

    This is a deterministic evaluation and does not call an LLM.

    Parameters
    ----------
    retrieved_documents:
        Documents returned by the Retriever.

    expected_keywords:
        Keywords that should ideally appear in the retrieved context.

    expected_sources:
        Source documents that are expected to be retrieved.

    Returns
    -------
    dict
        Retrieval evaluation metrics.
    """

    _validate_documents(retrieved_documents)

    expected_keywords = expected_keywords or []
    expected_sources = expected_sources or []

    # ---------------------------------------------------------
    # 1. Basic retrieval check
    # ---------------------------------------------------------

    documents_retrieved = len(retrieved_documents)

    if documents_retrieved == 0:
        return {
            "documents_retrieved": 0,
            "retrieval_success": False,
            "source_match_score": 0.0,
            "keyword_coverage_score": 0.0,
            "overall_score": 0.0,
        }

    # ---------------------------------------------------------
    # 2. Combine retrieved context
    # ---------------------------------------------------------

    retrieved_text = " ".join(
        _normalize_text(
            document.get("text", "")
        )
        for document in retrieved_documents
    )

    retrieved_sources = {
        str(document.get("source", "")).strip()
        for document in retrieved_documents
        if document.get("source")
    }

    # ---------------------------------------------------------
    # 3. Keyword coverage
    # ---------------------------------------------------------

    if expected_keywords:
        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if _normalize_text(keyword) in retrieved_text
        ]

        keyword_coverage_score = (
            len(matched_keywords)
            / len(expected_keywords)
        )
    else:
        matched_keywords = []
        keyword_coverage_score = 1.0

    # ---------------------------------------------------------
    # 4. Expected source matching
    # ---------------------------------------------------------

    if expected_sources:
        matched_sources = [
            source
            for source in expected_sources
            if source in retrieved_sources
        ]

        source_match_score = (
            len(matched_sources)
            / len(expected_sources)
        )
    else:
        matched_sources = []
        source_match_score = 1.0

    # ---------------------------------------------------------
    # 5. Overall retrieval score
    # ---------------------------------------------------------

    if expected_keywords and expected_sources:
        overall_score = (
            keyword_coverage_score
            + source_match_score
        ) / 2

    elif expected_keywords:
        overall_score = keyword_coverage_score

    elif expected_sources:
        overall_score = source_match_score

    else:
        overall_score = 1.0

    return {
        "documents_retrieved": documents_retrieved,
        "retrieval_success": True,
        "matched_keywords": matched_keywords,
        "expected_keywords": expected_keywords,
        "keyword_coverage_score": keyword_coverage_score,
        "matched_sources": matched_sources,
        "expected_sources": expected_sources,
        "source_match_score": source_match_score,
        "overall_score": overall_score,
    }