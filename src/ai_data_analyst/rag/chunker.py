def chunk_document(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> list[str]:
    """
    Split a document into paragraph-aware overlapping chunks.

    chunk_size:
        Maximum approximate number of characters per chunk.

    chunk_overlap:
        Approximate number of characters to preserve
        from the previous chunk.
    """

    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        candidate = (
            f"{current_chunk}\n\n{paragraph}"
            if current_chunk
            else paragraph
        )

        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)

        words = current_chunk.split()

        overlap_words = []
        overlap_length = 0

        for word in reversed(words):
            if overlap_length + len(word) + 1 > chunk_overlap:
                break

            overlap_words.insert(0, word)
            overlap_length += len(word) + 1

        overlap_text = " ".join(overlap_words)

        current_chunk = (
            f"{overlap_text}\n\n{paragraph}"
            if overlap_text
            else paragraph
        )

    if current_chunk:
        chunks.append(current_chunk)

    return chunks