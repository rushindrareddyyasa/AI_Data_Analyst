import pandas as pd


def get_dataset_schema(
    df: pd.DataFrame,
    table_name: str = "dataset",
) -> str:
    """
    Generate a SQL schema dynamically from a DataFrame.
    """

    if df.empty:
        raise ValueError("Dataset cannot be empty.")

    if not table_name:
        raise ValueError("Table name cannot be empty.")

    lines = [
        f"TABLE: {table_name}",
        "",
        "COLUMNS:",
        "",
    ]

    for column in df.columns:

        dtype = df[column].dtype

        if pd.api.types.is_datetime64_any_dtype(dtype):
            sql_type = "DATETIME"

        elif pd.api.types.is_integer_dtype(dtype):
            sql_type = "INTEGER"

        elif pd.api.types.is_float_dtype(dtype):
            sql_type = "REAL"

        elif pd.api.types.is_bool_dtype(dtype):
            sql_type = "BOOLEAN"

        else:
            sql_type = "TEXT"

        lines.append(
            f"{column:<30} {sql_type}"
        )

    return "\n".join(lines)