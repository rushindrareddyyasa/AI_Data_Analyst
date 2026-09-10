import pandas as pd
from sqlalchemy import create_engine


def create_database(
    df: pd.DataFrame,
    database_path: str = "data/processed/data.db",
    table_name: str = "dataset",
):
    if df.empty:
        raise ValueError("Dataset cannot be empty.")

    if not table_name:
        raise ValueError("Table name cannot be empty.")

    if database_path == ":memory:":
        engine = create_engine("sqlite:///:memory:")
    else:
        engine = create_engine(f"sqlite:///{database_path}")

    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
    )

    return engine