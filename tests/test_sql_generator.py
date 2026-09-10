from unittest.mock import patch

from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.sql.generator import generate_sql
from ai_data_analyst.sql.schema import get_dataset_schema


def test_generate_sql():

    df = load_csv("data/raw/sales.csv")
    schema = get_dataset_schema(df)

    question = "Which region generated the highest sales?"

    expected_sql = """
    SELECT region, SUM(sales) AS total_sales
    FROM dataset
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