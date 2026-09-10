import pandas as pd
from sqlalchemy import text


def execute_sql(engine, sql: str) -> pd.DataFrame:
    """
    Execute a validated SQL query and return the result
    as a pandas DataFrame.
    """

    with engine.connect() as connection:

        result = connection.execute(
            text(sql)
        )

        df = pd.DataFrame(
            result.fetchall(),
            columns=result.keys()
        )

    return df