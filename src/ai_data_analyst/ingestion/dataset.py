from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


def load_dataset(file_source) -> pd.DataFrame:
    """
    Load a CSV or Excel dataset.

    Supports:
    - Local file paths
    - Streamlit UploadedFile objects
    """

    # ---------------------------------------------
    # Determine file extension
    # ---------------------------------------------

    if isinstance(file_source, (str, Path)):

        path = Path(file_source)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {file_source}"
            )

        extension = path.suffix.lower()

    else:

        # Streamlit UploadedFile
        extension = Path(
            file_source.name
        ).suffix.lower()


    # ---------------------------------------------
    # Load CSV
    # ---------------------------------------------

    if extension == ".csv":

        encodings = [
            "utf-8",
            "utf-8-sig",
            "cp1252",
            "latin1",
        ]

        last_error = None

        for encoding in encodings:

            try:

                if isinstance(
                    file_source,
                    (str, Path)
                ):

                    df = pd.read_csv(
                        file_source,
                        encoding=encoding,
                    )

                else:

                    file_source.seek(0)

                    df = pd.read_csv(
                        file_source,
                        encoding=encoding,
                    )

                break

            except UnicodeDecodeError as error:

                last_error = error

        else:

            raise ValueError(
                "Unable to decode CSV using supported "
                f"encodings: {encodings}. "
                f"Last error: {last_error}"
            )


    # ---------------------------------------------
    # Load Excel
    # ---------------------------------------------

    elif extension == ".xlsx":

        if isinstance(
            file_source,
            (str, Path)
        ):

            df = pd.read_excel(
                file_source
            )

        else:

            file_source.seek(0)

            df = pd.read_excel(
                file_source
            )


    # ---------------------------------------------
    # Unsupported file
    # ---------------------------------------------

    else:

        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {SUPPORTED_EXTENSIONS}"
        )


    # ---------------------------------------------
    # Automatically detect date/time columns
    # ---------------------------------------------

    for column in df.columns:

        if (
            "date" in column.lower()
            or "time" in column.lower()
        ):

            converted = pd.to_datetime(
                df[column],
                errors="coerce",
            )

            if converted.notna().mean() >= 0.8:

                df[column] = converted


    return df