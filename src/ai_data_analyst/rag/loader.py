from pathlib import Path

from pypdf import PdfReader


SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".pdf"}


def _sanitize_text(text: str) -> str:
    """
    Remove or replace malformed Unicode characters
    so the text can safely be encoded as UTF-8.
    """
    if not text:
        return ""

    return text.encode("utf-8", errors="replace").decode("utf-8")


def _extract_pdf_text(file_source) -> str:
    """
    Extract text from a text-based PDF.
    """
    reader = PdfReader(file_source)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            text = _sanitize_text(text)
            pages.append(text)

    return "\n\n".join(pages)


def load_documents(directory: str) -> list[dict]:
    """
    Load TXT and PDF documents from a directory.
    """
    documents = []

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Document directory not found: {directory}"
        )

    for file_path in directory_path.iterdir():

        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
            continue

        if extension == ".txt":
            text = file_path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        elif extension == ".pdf":
            text = _extract_pdf_text(file_path)

        else:
            continue

        text = _sanitize_text(text)

        if not text.strip():
            continue

        documents.append(
            {
                "source": file_path.name,
                "text": text,
            }
        )

    return documents


def load_uploaded_document(uploaded_file) -> dict:
    """
    Load a user-uploaded TXT or PDF document.
    """
    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type: {extension}. "
            f"Supported document types: {SUPPORTED_DOCUMENT_EXTENSIONS}"
        )

    uploaded_file.seek(0)

    if extension == ".txt":

        text = uploaded_file.read()

        if isinstance(text, bytes):
            text = text.decode(
                "utf-8",
                errors="replace",
            )

    elif extension == ".pdf":

        text = _extract_pdf_text(uploaded_file)

    else:
        raise ValueError(
            f"Unsupported document type: {extension}"
        )

    text = _sanitize_text(text)

    if not text.strip():
        raise ValueError(
            f"Document '{uploaded_file.name}' "
            "does not contain extractable text."
        )

    return {
        "source": uploaded_file.name,
        "text": text,
    }