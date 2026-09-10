import pandas as pd

from ai_data_analyst.analysis.analyzer import (
    group_by_analysis,
    filtered_aggregation,
)


def execute_analysis(
    df: pd.DataFrame,
    decision
):
    """
    Execute a Python/Pandas analysis based on
    the router decision.

    Returns:
        Scalar value or pandas DataFrame.
    """

    # --------------------------------
    # SUPPORTED OPERATIONS
    # --------------------------------

    supported_operations = {
        "mean",
        "median",
        "sum",
        "min",
        "max",
        "count",
        "std",
    }

    operation = decision.operation

    if operation not in supported_operations:
        raise ValueError(
            f"Unsupported analysis operation: {operation}"
        )

    # --------------------------------
    # VALUE COLUMN
    # --------------------------------

    if not decision.value_column:
        raise ValueError(
            "Python analysis requires value_column."
        )

    if decision.value_column not in df.columns:
        raise ValueError(
            f"Column not found: {decision.value_column}"
        )

    # --------------------------------
    # FILTERED ANALYSIS
    # --------------------------------

    if decision.filter_column:

        if decision.filter_column not in df.columns:
            raise ValueError(
                f"Column not found: {decision.filter_column}"
            )

        if decision.filter_value is None:
            raise ValueError(
                "filter_value is required when "
                "filter_column is provided."
            )

        result = filtered_aggregation(
            df=df,
            filter_column=decision.filter_column,
            filter_value=decision.filter_value,
            value_column=decision.value_column,
            operation=operation,
        )

        return result

    # --------------------------------
    # GROUPED ANALYSIS
    # --------------------------------

    if decision.group_column:

        if decision.group_column not in df.columns:
            raise ValueError(
                f"Column not found: {decision.group_column}"
            )

        result = group_by_analysis(
            df=df,
            group_column=decision.group_column,
            value_column=decision.value_column,
            operation=operation,
        )

        return result

    # --------------------------------
    # SIMPLE AGGREGATION
    # --------------------------------

    result = getattr(
        df[decision.value_column],
        operation,
    )()

    return result