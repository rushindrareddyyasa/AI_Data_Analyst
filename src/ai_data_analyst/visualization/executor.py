import pandas as pd

from ai_data_analyst.visualization.charts import (
    create_bar_chart,
    create_line_chart,
    create_pie_chart,
    create_scatter_chart,
)


# ==================================================
# GENERATE VISUALIZATION CODE
# ==================================================

def generate_visualization_code(
    decision,
) -> str:
    """
    Generate a readable representation of the
    Pandas/Matplotlib logic used for visualization.

    The code is generated from the validated router
    decision and represents the logic executed by the
    visualization engine.
    """

    chart_type = decision.chart_type

    # ----------------------------------------------
    # BAR CHART
    # ----------------------------------------------

    if chart_type == "bar":

        if not decision.group_column:
            raise ValueError(
                "Bar chart requires group_column."
            )

        if not decision.value_column:
            raise ValueError(
                "Bar chart requires value_column."
            )

        return (
            f"grouped_df = (\n"
            f"    df.groupby('{decision.group_column}')"
            f"['{decision.value_column}']"
            f".agg('{decision.operation}')"
            f".reset_index()\n"
            f")\n\n"
            f"fig = create_bar_chart(\n"
            f"    grouped_df,\n"
            f"    '{decision.group_column}',\n"
            f"    '{decision.value_column}',\n"
            f"    title='{decision.operation.title()} "
            f"{decision.value_column} by "
            f"{decision.group_column}',\n"
            f")"
        )

    # ----------------------------------------------
    # PIE CHART
    # ----------------------------------------------

    if chart_type == "pie":

        if not decision.group_column:
            raise ValueError(
                "Pie chart requires group_column."
            )

        if not decision.value_column:
            raise ValueError(
                "Pie chart requires value_column."
            )

        return (
            f"grouped_df = (\n"
            f"    df.groupby('{decision.group_column}')"
            f"['{decision.value_column}']"
            f".agg('{decision.operation}')"
            f".reset_index()\n"
            f")\n\n"
            f"fig = create_pie_chart(\n"
            f"    grouped_df,\n"
            f"    '{decision.group_column}',\n"
            f"    '{decision.value_column}',\n"
            f"    title='{decision.value_column} "
            f"Distribution by "
            f"{decision.group_column}',\n"
            f")"
        )

    # ----------------------------------------------
    # SCATTER PLOT
    # ----------------------------------------------

    if chart_type == "scatter":

        if not decision.x_column:
            raise ValueError(
                "Scatter plot requires x_column."
            )

        if not decision.y_column:
            raise ValueError(
                "Scatter plot requires y_column."
            )

        return (
            f"fig = create_scatter_chart(\n"
            f"    df,\n"
            f"    '{decision.x_column}',\n"
            f"    '{decision.y_column}',\n"
            f"    title='{decision.y_column} vs "
            f"{decision.x_column}',\n"
            f")"
        )

    # ----------------------------------------------
    # LINE CHART
    # ----------------------------------------------

    if chart_type == "line":

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
                "Line chart requires time_granularity."
            )

        code_lines = [
            "data = df.copy()",
            "",
            f"data['{decision.date_column}'] = "
            f"pd.to_datetime(",
            f"    data['{decision.date_column}'],",
            "    errors='coerce',",
            ")",
        ]

        # ------------------------------------------
        # YEAR FILTER
        # ------------------------------------------

        if decision.filter_year is not None:

            code_lines.extend(
                [
                    "",
                    f"data = data[",
                    f"    data['{decision.date_column}']"
                    f".dt.year == {decision.filter_year}",
                    "]",
                ]
            )

        # ------------------------------------------
        # TIME GRANULARITY
        # ------------------------------------------

        if decision.time_granularity == "month":

            code_lines.extend(
                [
                    "",
                    "data['time_period'] = (",
                    f"    data['{decision.date_column}']",
                    ".dt.to_period('M')",
                    ".dt.to_timestamp()",
                    ")",
                ]
            )

        elif decision.time_granularity == "year":

            code_lines.extend(
                [
                    "",
                    "data['time_period'] = (",
                    f"    data['{decision.date_column}']",
                    ".dt.to_period('Y')",
                    ".dt.to_timestamp()",
                    ")",
                ]
            )

        elif decision.time_granularity == "day":

            code_lines.extend(
                [
                    "",
                    "data['time_period'] = (",
                    f"    data['{decision.date_column}']",
                    ".dt.floor('D')",
                    ")",
                ]
            )

        else:

            raise ValueError(
                f"Unsupported time granularity: "
                f"{decision.time_granularity}"
            )

        # ------------------------------------------
        # GROUP DATA
        # ------------------------------------------

        code_lines.extend(
            [
                "",
                "grouped_df = (",
                "    data.groupby('time_period')"
                f"['{decision.value_column}']",
                f".agg('{decision.operation}')",
                ".reset_index()",
                ")",
                "",
                "fig = create_line_chart(",
                "    grouped_df,",
                "    'time_period',",
                f"    '{decision.value_column}',",
                f"    title='{decision.operation.title()} "
                f"{decision.value_column} "
                f"by {decision.time_granularity}',",
                ")",
            ]
        )

        return "\n".join(code_lines)

    # ----------------------------------------------
    # UNKNOWN CHART TYPE
    # ----------------------------------------------

    raise ValueError(
        f"Unsupported chart type: {chart_type}"
    )


# ==================================================
# EXECUTE VISUALIZATION
# ==================================================

def execute_visualization(
    df: pd.DataFrame,
    decision,
):
    """
    Execute a visualization based on the router decision.

    Returns:
        matplotlib Figure
    """

    chart_type = decision.chart_type

    # ----------------------------------------------
    # BAR CHART
    # ----------------------------------------------

    if chart_type == "bar":

        if not decision.group_column:
            raise ValueError(
                "Bar chart requires group_column."
            )

        if not decision.value_column:
            raise ValueError(
                "Bar chart requires value_column."
            )

        grouped_df = (
            df.groupby(
                decision.group_column
            )[decision.value_column]
            .agg(decision.operation)
            .reset_index()
        )

        return create_bar_chart(
            grouped_df,
            decision.group_column,
            decision.value_column,
            title=(
                f"{decision.operation.title()} "
                f"{decision.value_column} by "
                f"{decision.group_column}"
            ),
        )

    # ----------------------------------------------
    # PIE CHART
    # ----------------------------------------------

    if chart_type == "pie":

        if not decision.group_column:
            raise ValueError(
                "Pie chart requires group_column."
            )

        if not decision.value_column:
            raise ValueError(
                "Pie chart requires value_column."
            )

        grouped_df = (
            df.groupby(
                decision.group_column
            )[decision.value_column]
            .agg(decision.operation)
            .reset_index()
        )

        return create_pie_chart(
            grouped_df,
            decision.group_column,
            decision.value_column,
            title=(
                f"{decision.value_column} "
                f"Distribution by "
                f"{decision.group_column}"
            ),
        )

    # ----------------------------------------------
    # SCATTER PLOT
    # ----------------------------------------------

    if chart_type == "scatter":

        if not decision.x_column:
            raise ValueError(
                "Scatter plot requires x_column."
            )

        if not decision.y_column:
            raise ValueError(
                "Scatter plot requires y_column."
            )

        return create_scatter_chart(
            df,
            decision.x_column,
            decision.y_column,
            title=(
                f"{decision.y_column} vs "
                f"{decision.x_column}"
            ),
        )

    # ----------------------------------------------
    # LINE CHART
    # ----------------------------------------------

    if chart_type == "line":

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
                "Line chart requires time_granularity."
            )

        data = df.copy()

        # ------------------------------------------
        # ENSURE DATE COLUMN IS DATETIME
        # ------------------------------------------

        data[decision.date_column] = pd.to_datetime(
            data[decision.date_column],
            errors="coerce",
        )

        # ------------------------------------------
        # FILTER BY YEAR
        # ------------------------------------------

        if decision.filter_year is not None:

            data = data[
                data[decision.date_column].dt.year
                == decision.filter_year
            ]

        # ------------------------------------------
        # MONTHLY
        # ------------------------------------------

        if decision.time_granularity == "month":

            data["time_period"] = (
                data[decision.date_column]
                .dt.to_period("M")
                .dt.to_timestamp()
            )

        # ------------------------------------------
        # YEARLY
        # ------------------------------------------

        elif decision.time_granularity == "year":

            data["time_period"] = (
                data[decision.date_column]
                .dt.to_period("Y")
                .dt.to_timestamp()
            )

        # ------------------------------------------
        # DAILY
        # ------------------------------------------

        elif decision.time_granularity == "day":

            data["time_period"] = (
                data[decision.date_column]
                .dt.floor("D")
            )

        else:

            raise ValueError(
                f"Unsupported time granularity: "
                f"{decision.time_granularity}"
            )

        # ------------------------------------------
        # GROUP DATA
        # ------------------------------------------

        grouped_df = (
            data.groupby("time_period")
            [decision.value_column]
            .agg(decision.operation)
            .reset_index()
        )

        return create_line_chart(
            grouped_df,
            "time_period",
            decision.value_column,
            title=(
                f"{decision.operation.title()} "
                f"{decision.value_column} "
                f"by {decision.time_granularity}"
            ),
        )

    # ----------------------------------------------
    # UNKNOWN CHART TYPE
    # ----------------------------------------------

    raise ValueError(
        f"Unsupported chart type: {chart_type}"
    )