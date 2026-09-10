from pathlib import Path


def load_documents(directory: str) -> list[dict]:
    """
    Load text documents from a directory.

    Returns:
        A list of dictionaries containing:
        - source: document filename
        - text: document content
    """

    documents = []

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Document directory not found: {directory}"
        )

    for file_path in directory_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "source": file_path.name,
                "text": text,
            }
        )

    return documents