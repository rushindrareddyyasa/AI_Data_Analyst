import json
from unittest.mock import patch

from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.sql.database import create_database
from ai_data_analyst.agent import run_agent


def test_agent():

    df = load_csv("data/raw/sales.csv")
    engine = create_database(df)

    questions = [
        "Which region generated the highest sales?",
        "What is the average profit?",
        "Show me total sales by region as a bar chart.",
        "Show monthly sales for 2025 as a line chart.",
        "Show the relationship between discount and profit.",
        "When is a region considered high-performing?",
    ]

    mock_responses = [
        {
            "tool": "sql",
            "operation": "sum",
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
            "x_column": "order_date",
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
            "tool": "rag",
            "operation": None,
            "group_column": None,
            "value_column": None,
            "filter_column": None,
            "filter_value": None,
            "chart_type": None,
            "x_column": None,
            "y_column": None,
            "date_column": None,
            "time_granularity": None,
            "filter_year": None,
        },
    ]

    for question, mock_response in zip(
        questions,
        mock_responses
    ):

        with patch(
            "ai_data_analyst.llm.router.client.models.generate_content"
        ) as mock_generate:

            mock_generate.return_value.text = json.dumps(
                mock_response
            )

            # Mock SQL generation
            with patch(
                "ai_data_analyst.sql.query_engine.generate_sql"
            ) as mock_sql:

                mock_sql.return_value = """
                SELECT region, SUM(sales) AS total_sales
                FROM dataset
                GROUP BY region
                ORDER BY total_sales DESC
                LIMIT 1;
                """.strip()

                # Mock SQL result explanation
                with patch(
                    "ai_data_analyst.sql.query_engine.explain_result"
                ) as mock_explain:

                    mock_explain.return_value = (
                        "The West region generated the highest sales."
                    )

                    # Mock Python result explanation
                    with patch(
                        "ai_data_analyst.llm.response.generate_response"
                    ) as mock_response_generate:

                        mock_response_generate.return_value = (
                            "The average profit is 1190.94."
                        )

                        # Mock RAG pipeline
                        with patch(
                            "ai_data_analyst.agent.RAGPipeline"
                        ) as mock_rag_pipeline:

                            mock_rag_pipeline.return_value.answer_with_sources.return_value = {
                                "answer": (
                                    "A region is considered "
                                    "high-performing when its "
                                    "quarterly sales exceed its "
                                    "assigned target by more "
                                    "than 10 percent."
                                ),
                                "sources": [
                                    "regional_sales_policy.txt"
                                ],
                            }

                            result = run_agent(
                                df,
                                engine,
                                question,
                            )

        assert result is not None
        assert result["tool"] == mock_response["tool"]
        assert result["answer"] is not None

        if result["tool"] == "rag":

            assert (
                "high-performing"
                in result["answer"]
            )

            assert result["sources"] == [
                "regional_sales_policy.txt"
            ]

        if result["figure"] is not None:
            result["figure"].clf()