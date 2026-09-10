from unittest.mock import patch

from ai_data_analyst.sql.generator import generate_sql
from ai_data_analyst.sql.schema import get_sales_schema


def test_generate_sql():

    schema = get_sales_schema()

    question = "Which region generated the highest sales?"

    expected_sql = """
    SELECT region, SUM(sales) AS total_sales
    FROM sales
    GROUP BY region
    ORDER BY total_sales DESC
    LIMIT 1;
    """.strip()

    with patch(
        "ai_data_analyst.sql.generator.generate_response"
    ) as mock_generate:

        mock_generate.return_value = expected_sql

        sql = generate_sql(
            schema,
            question
        )

        assert sql == expected_sql

        mock_generate.assert_called_once()