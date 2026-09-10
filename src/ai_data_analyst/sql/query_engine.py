import pandas as pd

from ai_data_analyst.sql.generator import generate_sql
from ai_data_analyst.sql.validator import validate_sql
from ai_data_analyst.sql.executor import execute_sql
from ai_data_analyst.sql.schema import get_dataset_schema
from ai_data_analyst.llm.response import explain_result


def answer_question(
    df: pd.DataFrame,
    engine,
    question: str,
):
    """
    Answer a user's question using the active dataset.
    """

    schema = get_dataset_schema(df)

    sql = generate_sql(
        schema=schema,
        question=question,
    )

    print("\n========== GENERATED SQL ==========\n")
    print(sql)

    if not validate_sql(sql):
        raise ValueError(
            "Generated SQL query is unsafe!"
        )

    result = execute_sql(
        engine=engine,
        sql=sql,
    )

    print("\n========== SQL RESULT ==========\n")
    print(result)

    answer = explain_result(
        question,
        result,
    )

    return answer