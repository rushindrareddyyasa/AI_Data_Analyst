import pandas as pd

from ai_data_analyst.llm.client import generate_response


def explain_result(
    question: str,
    result: pd.DataFrame
) -> str:
    """
    Convert a tabular analysis result into
    a natural-language business answer.
    """

    result_text = result.to_string(index=False)

    prompt = f"""
You are an AI Data Analyst.

USER QUESTION
-------------
{question}

ANALYSIS RESULT
---------------
{result_text}

INSTRUCTIONS
------------

Answer the user's question using only the
analysis result provided above.

Rules:

1. Give a clear and concise business answer.
2. Do not invent any values.
3. Do not perform additional calculations.
4. Mention important numbers when relevant.
5. Do not mention SQL, databases, DataFrames,
   pandas, or internal implementation.
6. Do not assume or invent currency symbols or units.
7. Use the numbers exactly as provided.
"""

    return generate_response(prompt)


def explain_scalar_result(
    question: str,
    result,
    operation: str,
    value_column: str
) -> str:
    """
    Convert a scalar analysis result into
    a natural-language business answer.
    """

    result_text = str(result)

    prompt = f"""
You are an AI Data Analyst.

USER QUESTION
-------------
{question}

ANALYSIS OPERATION
------------------
{operation}

ANALYZED COLUMN
---------------
{value_column}

ANALYSIS RESULT
---------------
{result_text}

INSTRUCTIONS
------------

Answer the user's question using only
the analysis result provided above.

Rules:

1. Give a clear and concise answer.
2. Do not invent any values.
3. Do not perform additional calculations.
4. Mention the result clearly.
5. Do not mention Python, pandas, DataFrames,
   SQL, databases, or internal implementation.
6. Do not assume or invent currency symbols or units.
7. Use the result exactly as provided.
"""

    return generate_response(prompt)