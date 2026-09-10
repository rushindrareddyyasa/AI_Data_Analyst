import pandas as pd


def calculate_summary(df: pd.DataFrame) -> dict:
    """
    Calculate basic statistical summaries for numerical columns.
    """

    numerical_df = df.select_dtypes(include="number")

    return {
        "mean": numerical_df.mean().to_dict(),
        "median": numerical_df.median().to_dict(),
        "min": numerical_df.min().to_dict(),
        "max": numerical_df.max().to_dict(),
        "std": numerical_df.std().to_dict(),
    }

def group_by_analysis(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    operation: str = "mean"
) -> pd.DataFrame:
    """
    Perform an aggregation on a numerical column
    grouped by a categorical column.

    Supported operations:
    mean, sum, min, max, median, count
    """

    supported_operations = {
        "mean": "mean",
        "sum": "sum",
        "min": "min",
        "max": "max",
        "median": "median",
        "count": "count",
    }

    if operation not in supported_operations:
        raise ValueError(
            f"Unsupported operation: {operation}. "
            f"Supported operations: {list(supported_operations.keys())}"
        )

    if group_column not in df.columns:
        raise ValueError(f"Column not found: {group_column}")

    if value_column not in df.columns:
        raise ValueError(f"Column not found: {value_column}")

    result = (
        df.groupby(group_column)[value_column]
        .agg(supported_operations[operation])
        .reset_index()
    )

    result.columns = [group_column, f"{operation}_{value_column}"]

    return result

def filtered_aggregation(
    df: pd.DataFrame,
    filter_column: str,
    filter_value,
    value_column: str,
    operation: str = "mean"
):
    """
    Filter the dataset and perform an aggregation
    on the selected numerical column.
    """

    supported_operations = {
        "mean": "mean",
        "sum": "sum",
        "min": "min",
        "max": "max",
        "median": "median",
        "count": "count",
    }

    if operation not in supported_operations:
        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    if filter_column not in df.columns:
        raise ValueError(
            f"Column not found: {filter_column}"
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Column not found: {value_column}"
        )

    filtered_df = df[
        df[filter_column] == filter_value
    ]

    if filtered_df.empty:
        return None

    result = getattr(
        filtered_df[value_column],
        supported_operations[operation]
    )()

    return result