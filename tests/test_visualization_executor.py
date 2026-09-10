from unittest.mock import patch

from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.llm.router import route_question
from ai_data_analyst.visualization.executor import execute_visualization


def test_visualization_executor():

    df = load_csv("data/raw/sales.csv")

    questions = [
        "Show me total sales by region as a bar chart.",
        "Show monthly sales for 2025 as a line chart.",
        "Show the relationship between discount and profit.",
        "Show sales distribution by region as a pie chart.",
    ]

    mock_responses = [
        {
            "tool": "visualization",
            "operation": "sum",
            "group_column": "region",
            "value_column": "sales",
            "filter_column": None,
            "filter_value": None,
            "chart_type": "bar",
            "x_column": None,
            "y_column": None,
            "date_column": None,
            "time_granularity": None,
            "filter_year": None,
        },
        {
            "tool": "visualization",
            "operation": "sum",
            "group_column": None,
            "value_column": "sales",
            "filter_column": None,
            "filter_value": None,
            "chart_type": "line",
            "x_column": None,
            "y_column": None,
            "date_column": "order_date",
            "time_granularity": "month",
            "filter_year": 2025,
        },
        {
            "tool": "visualization",
            "operation": None,
            "group_column": None,
            "value_column": None,
            "filter_column": None,
            "filter_value": None,
            "chart_type": "scatter",
            "x_column": "discount",
            "y_column": "profit",
            "date_column": None,
            "time_granularity": None,
            "filter_year": None,
        },
        {
            "tool": "visualization",
            "operation": "sum",
            "group_column": "region",
            "value_column": "sales",
            "filter_column": None,
            "filter_value": None,
            "chart_type": "pie",
            "x_column": None,
            "y_column": None,
            "date_column": None,
            "time_granularity": None,
            "filter_year": None,
        },
    ]

    for question, mock_response in zip(
        questions,
        mock_responses,
    ):

        with patch(
            "ai_data_analyst.llm.router.client.models.generate_content"
        ) as mock_generate:

            import json

            mock_generate.return_value.text = json.dumps(
                mock_response
            )

            decision = route_question(question)

            figure = execute_visualization(
                df,
                decision
            )

            assert figure is not None

            # Close the figure after testing
            figure.clf()