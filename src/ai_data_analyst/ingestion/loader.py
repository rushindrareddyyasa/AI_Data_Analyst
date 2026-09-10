import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file and automatically detect date columns.
    """

    df = pd.read_csv(file_path)

    # Try to convert columns containing date/time information
    for column in df.columns:
        if "date" in column.lower() or "time" in column.lower():
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    return df