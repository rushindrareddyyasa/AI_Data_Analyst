from unittest.mock import patch

from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.sql.database import create_database
from ai_data_analyst.sql.query_engine import answer_question


def test_answer_question():

    # Load dataset
    df = load_csv("data/raw/sales.csv")

    # Create SQLite database
    engine = create_database(df)

    question = "Which region generated the highest sales?"

    expected_sql = """
    SELECT region, SUM(sales) AS total_sales
    FROM sales
    GROUP BY region
    ORDER BY total_sales DESC
    LIMIT 1;
    """.strip()

    expected_answer = (
        "The West region generated the highest sales, "
        "with a total of 4835478.19."
    )

    # Mock SQL generation and final answer generation
    with patch(
        "ai_data_analyst.sql.query_engine.generate_sql"
    ) as mock_generate_sql, patch(
        "ai_data_analyst.sql.query_engine.explain_result"
    ) as mock_explain_result:

        mock_generate_sql.return_value = expected_sql
        mock_explain_result.return_value = expected_answer

        answer = answer_question(
            engine,
            question
        )

        # Verify final answer
        assert answer == expected_answer

        # Verify SQL generation was called
        mock_generate_sql.assert_called_once()

        # Verify result explanation was called
        mock_explain_result.assert_called_once()