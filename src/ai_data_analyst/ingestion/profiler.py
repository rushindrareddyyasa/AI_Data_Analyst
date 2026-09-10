import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a detailed profile of the dataset.
    """

    # Basic dataset information
    rows = df.shape[0]
    columns = df.shape[1]

    # Missing value analysis
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / rows * 100).round(2)

    # Duplicate analysis
    duplicate_rows = int(df.duplicated().sum())

    # Column classification
    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns.tolist()

    date_columns = df.select_dtypes(
        include="datetime"
    ).columns.tolist()

    # General column information
    column_information = {}

    for column in df.columns:

        column_information[column] = {
            "data_type": str(df[column].dtype),
            "missing_values": int(missing_values[column]),
            "missing_percentage": float(missing_percentage[column]),
            "unique_values": int(df[column].nunique()),
        }

    # Numerical statistics
    numerical_statistics = {}

    for column in numerical_columns:

        numerical_statistics[column] = {
            "min": float(df[column].min()),
            "max": float(df[column].max()),
            "mean": float(df[column].mean()),
            "median": float(df[column].median()),
        }

    # Date range
    date_ranges = {}

    for column in date_columns:

        date_ranges[column] = {
            "min": str(df[column].min()),
            "max": str(df[column].max()),
        }

    profile = {
        "rows": rows,
        "columns": columns,
        "column_names": df.columns.tolist(),
        "duplicate_rows": duplicate_rows,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "column_information": column_information,
        "numerical_statistics": numerical_statistics,
        "date_ranges": date_ranges,
    }

    return profile