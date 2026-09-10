import pandas as pd


def filter_data(
    df: pd.DataFrame,
    column: str,
    value
) -> pd.DataFrame:
    """
    Filter the dataset based on an exact column value.
    """

    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    filtered_df = df[df[column] == value].copy()

    return filtered_df