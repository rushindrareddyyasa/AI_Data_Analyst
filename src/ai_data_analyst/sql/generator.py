from ai_data_analyst.llm.client import generate_response


def generate_sql(schema: str, question: str) -> str:
    """
    Generate a SQL query from a natural language question.
    """

    prompt = f"""
You are an expert SQL analyst.

You have access to a SQLite database.

DATABASE SCHEMA
---------------

{schema}

USER QUESTION
-------------

{question}

INSTRUCTIONS
------------

Generate a SQL query that answers the user's question.

Rules:
1. Return ONLY the SQL query.
2. Do not use markdown code blocks.
3. Use SQLite-compatible SQL.
4. Use only tables and columns that exist in the schema.
5. Do not modify the database.
6. Only generate SELECT queries.
7. Use clear aliases for calculated values.
8. When answering questions involving highest, lowest, maximum,
   minimum, top, or bottom, include both the relevant entity and
   the calculated metric in the SELECT statement.
"""

    response = generate_response(prompt)

    # Remove markdown fences if the model returns them
    sql = response.strip()

    if sql.startswith("```"):
        sql = sql.replace("```sql", "")
        sql = sql.replace("```", "")
        sql = sql.strip()

    return sql