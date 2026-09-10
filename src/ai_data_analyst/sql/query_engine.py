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
    Generate, validate, execute, and explain a SQL query.

    Returns both the final answer and the SQL query
    used to produce that answer.
    """

    # ==================================================
    # GENERATE DYNAMIC SCHEMA
    # ==================================================

    schema = get_dataset_schema(df)

    # ==================================================
    # GENERATE SQL
    # ==================================================

    sql = generate_sql(
        schema=schema,
        question=question,
    )

    print(
        "\n========== GENERATED SQL ==========\n"
    )

    print(sql)

    # ==================================================
    # VALIDATE SQL
    # ==================================================

    if not validate_sql(sql):

        raise ValueError(
            "Generated SQL query is unsafe!"
        )

    # ==================================================
    # EXECUTE SQL
    # ==================================================

    result = execute_sql(
        engine=engine,
        sql=sql,
    )

    print(
        "\n========== SQL RESULT ==========\n"
    )

    print(result)

    # ==================================================
    # EXPLAIN RESULT
    # ==================================================

    answer = explain_result(
        question,
        result,
    )

    # ==================================================
    # RETURN EVERYTHING
    # ==================================================

    return {
        "answer": answer,
        "sql": sql,
        "result": result,
    }