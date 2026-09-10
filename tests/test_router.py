import json
from unittest.mock import patch

from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.llm.router import route_question
df = load_csv("data/raw/sales.csv")

def test_router_questions():

    mock_responses = [
        {
            "tool": "sql",
            "operation": "max",
            "group_column": "region",
            "value_column": "sales",
            "filter_column": None,
            "filter_value": None,
            "chart_type": None,
            "x_column": None,
            "y_column": None,
            "date_column": None,
            "time_granularity": None,
            "filter_year": None,
        },
        {
            "tool": "python",
            "operation": "mean",
            "group_column": None,
            "value_column": "profit",
            "filter_column": None,
            "filter_value": None,
            "chart_type": None,
            "x_column": None,
            "y_column": None,
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
    ]

    questions = [
        "Which region generated the highest sales?",
        "What is the average profit?",
        "Show me total sales by region as a bar chart.",
        "Show monthly sales for 2025 as a line chart.",
        "Show the relationship between discount and profit.",
    ]

    for question, mock_response in zip(
        questions,
        mock_responses,
    ):

        with patch(
            "ai_data_analyst.llm.router.client.models.generate_content"
        ) as mock_generate:

            mock_generate.return_value.text = json.dumps(mock_response)

            decision = route_question(question,df)

            assert decision.tool == mock_response["tool"]
            assert decision.operation == mock_response["operation"]
            assert decision.value_column == mock_response["value_column"]
            assert decision.chart_type == mock_response["chart_type"]