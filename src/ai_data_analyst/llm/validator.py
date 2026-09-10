import pandas as pd

from ai_data_analyst.llm.router import RouteDecision


SUPPORTED_TOOLS = {
    "sql",
    "python",
    "visualization",
    "rag",
}

SUPPORTED_OPERATIONS = {
    "mean",
    "median",
    "std",
    "sum",
    "min",
    "max",
    "count",
}

SUPPORTED_CHART_TYPES = {
    "bar",
    "line",
    "pie",
    "scatter",
}

SUPPORTED_TIME_GRANULARITIES = {
    "day",
    "month",
    "year",
}


def validate_route(
    decision: RouteDecision,
    df: pd.DataFrame,
) -> bool:
    """
    Validate the routing decision produced by the LLM.

    This prevents the router from selecting unsupported
    tools, columns, operations, chart types, or
    time granularities.
    """

    # --------------------------------
    # TOOL VALIDATION
    # --------------------------------

    if decision.tool not in SUPPORTED_TOOLS:
        raise ValueError(
            f"Unsupported tool: {decision.tool}"
        )

    # --------------------------------
    # OPERATION VALIDATION
    # --------------------------------

    if decision.operation is not None:
        if decision.operation not in SUPPORTED_OPERATIONS:
            raise ValueError(
                f"Unsupported operation: "
                f"{decision.operation}"
            )

    # --------------------------------
    # COLUMN VALIDATION
    # --------------------------------

    if decision.value_column is not None:
        if decision.value_column not in df.columns:
            raise ValueError(
                f"Invalid value_column: "
                f"{decision.value_column}"
            )

    if decision.group_column is not None:
        if decision.group_column not in df.columns:
            raise ValueError(
                f"Invalid group_column: "
                f"{decision.group_column}"
            )

    if decision.filter_column is not None:
        if decision.filter_column not in df.columns:
            raise ValueError(
                f"Invalid filter_column: "
                f"{decision.filter_column}"
            )

    if decision.x_column is not None:
        if decision.x_column not in df.columns:
            raise ValueError(
                f"Invalid x_column: "
                f"{decision.x_column}"
            )

    if decision.y_column is not None:
        if decision.y_column not in df.columns:
            raise ValueError(
                f"Invalid y_column: "
                f"{decision.y_column}"
            )

    # --------------------------------
    # DATE COLUMN VALIDATION
    # --------------------------------

    if decision.date_column is not None:

        if decision.date_column not in df.columns:
            raise ValueError(
                f"Invalid date_column: "
                f"{decision.date_column}"
            )

        if not pd.api.types.is_datetime64_any_dtype(
            df[decision.date_column]
        ):
            raise ValueError(
                f"Date column must contain datetime "
                f"values: {decision.date_column}"
            )

    # --------------------------------
    # CHART VALIDATION
    # --------------------------------

    if decision.chart_type is not None:

        if decision.chart_type not in SUPPORTED_CHART_TYPES:
            raise ValueError(
                f"Unsupported chart type: "
                f"{decision.chart_type}"
            )

    # --------------------------------
    # TIME GRANULARITY VALIDATION
    # --------------------------------

    if decision.time_granularity is not None:

        if (
            decision.time_granularity
            not in SUPPORTED_TIME_GRANULARITIES
        ):
            raise ValueError(
                f"Unsupported time granularity: "
                f"{decision.time_granularity}"
            )

    # --------------------------------
    # PYTHON TOOL REQUIREMENTS
    # --------------------------------

    if decision.tool == "python":

        if not decision.operation:
            raise ValueError(
                "Python analysis requires an operation."
            )

        if not decision.value_column:
            raise ValueError(
                "Python analysis requires "
                "a value_column."
            )

    # --------------------------------
    # VISUALIZATION REQUIREMENTS
    # --------------------------------

    if decision.tool == "visualization":

        if not decision.chart_type:
            raise ValueError(
                "Visualization requires chart_type."
            )

        # Bar / Pie
        if decision.chart_type in {
            "bar",
            "pie",
        }:

            if not decision.group_column:
                raise ValueError(
                    f"{decision.chart_type} chart "
                    "requires group_column."
                )

            if not decision.value_column:
                raise ValueError(
                    f"{decision.chart_type} chart "
                    "requires value_column."
                )

        # Scatter
        if decision.chart_type == "scatter":

            if not decision.x_column:
                raise ValueError(
                    "Scatter plot requires x_column."
                )

            if not decision.y_column:
                raise ValueError(
                    "Scatter plot requires y_column."
                )

        # Line
        if decision.chart_type == "line":

            if not decision.date_column:
                raise ValueError(
                    "Line chart requires date_column."
                )

            if not decision.value_column:
                raise ValueError(
                    "Line chart requires value_column."
                )

            if not decision.time_granularity:
                raise ValueError(
                    "Line chart requires "
                    "time_granularity."
                )

    # --------------------------------
    # RAG REQUIREMENTS
    # --------------------------------

    if decision.tool == "rag":

        # RAG does not require dataset columns,
        # numerical operations, or chart information.

        return True

    # --------------------------------
    # VALID ROUTE
    # --------------------------------

    return True