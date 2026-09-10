import pandas as pd

from ai_data_analyst.analysis.analyzer import (
    group_by_analysis,
    filtered_aggregation,
)


# ==================================================
# SUPPORTED OPERATIONS
# ==================================================

SUPPORTED_OPERATIONS = {
    "mean",
    "median",
    "sum",
    "min",
    "max",
    "count",
    "std",
}


# ==================================================
# GENERATE PYTHON CODE
# ==================================================

def generate_analysis_code(
    decision,
) -> str:
    """
    Generate a readable representation of the
    Pandas operation selected by the router.

    This code is generated from the validated
    routing decision. It represents the logic
    executed by the analysis engine.
    """

    operation = decision.operation
    value_column = decision.value_column

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported analysis operation: {operation}"
        )

    if not value_column:
        raise ValueError(
            "Python analysis requires value_column."
        )

    # ----------------------------------------------
    # FILTERED ANALYSIS
    # ----------------------------------------------

    if decision.filter_column:

        filter_column = decision.filter_column
        filter_value = decision.filter_value

        return (
            f"result = df[\n"
            f"    df['{filter_column}'] == "
            f"{filter_value!r}\n"
            f"]['{value_column}'].{operation}()"
        )

    # ----------------------------------------------
    # GROUPED ANALYSIS
    # ----------------------------------------------

    if decision.group_column:

        group_column = decision.group_column

        return (
            f"result = (\n"
            f"    df.groupby('{group_column}')"
            f"['{value_column}']"
            f".{operation}()"
            f".reset_index()\n"
            f")"
        )

    # ----------------------------------------------
    # SIMPLE AGGREGATION
    # ----------------------------------------------

    return (
        f"result = df['{value_column}'].{operation}()"
    )


# ==================================================
# EXECUTE ANALYSIS
# ==================================================

def execute_analysis(
    df: pd.DataFrame,
    decision,
):
    """
    Execute a Python/Pandas analysis based on
    the router decision.

    Returns:
        Scalar value or pandas DataFrame.
    """

    # ----------------------------------------------
    # VALIDATE OPERATION
    # ----------------------------------------------

    operation = decision.operation

    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"Unsupported analysis operation: {operation}"
        )

    # ----------------------------------------------
    # VALIDATE VALUE COLUMN
    # ----------------------------------------------

    if not decision.value_column:
        raise ValueError(
            "Python analysis requires value_column."
        )

    if decision.value_column not in df.columns:
        raise ValueError(
            f"Column not found: {decision.value_column}"
        )

    # ----------------------------------------------
    # FILTERED ANALYSIS
    # ----------------------------------------------

    if decision.filter_column:

        if decision.filter_column not in df.columns:
            raise ValueError(
                f"Column not found: "
                f"{decision.filter_column}"
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

    # ----------------------------------------------
    # GROUPED ANALYSIS
    # ----------------------------------------------

    if decision.group_column:

        if decision.group_column not in df.columns:
            raise ValueError(
                f"Column not found: "
                f"{decision.group_column}"
            )

        result = group_by_analysis(
            df=df,
            group_column=decision.group_column,
            value_column=decision.value_column,
            operation=operation,
        )

        return result

    # ----------------------------------------------
    # SIMPLE AGGREGATION
    # ----------------------------------------------

    result = getattr(
        df[decision.value_column],
        operation,
    )()

    return result