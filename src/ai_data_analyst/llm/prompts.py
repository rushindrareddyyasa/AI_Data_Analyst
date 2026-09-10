def create_dataset_prompt(profile: dict, user_question: str) -> str:
    """
    Create a prompt containing dataset metadata and the user's question.
    """

    prompt = f"""
You are an AI Data Analyst.

You are analyzing a sales dataset.

DATASET INFORMATION
-------------------

Rows: {profile['rows']}
Columns: {profile['columns']}

Numerical Columns:
{profile['numerical_columns']}

Categorical Columns:
{profile['categorical_columns']}

Date Columns:
{profile['date_columns']}

Duplicate Rows:
{profile['duplicate_rows']}

COLUMN INFORMATION
------------------

{profile['column_information']}

NUMERICAL STATISTICS
--------------------

{profile['numerical_statistics']}

DATE RANGES
-----------

{profile['date_ranges']}


USER QUESTION
-------------

{user_question}


INSTRUCTIONS
------------

Answer the user's question based only on the dataset information
provided above.

If the available information is insufficient to answer the question,
clearly say that more analysis of the actual dataset is required.

Do not invent values.
"""

    return prompt