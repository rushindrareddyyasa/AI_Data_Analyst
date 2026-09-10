from dataclasses import dataclass
from typing import Optional
import json

import pandas as pd

from ai_data_analyst.sql.schema import get_dataset_schema
from ai_data_analyst.llm.client import client


# ==================================================
# ROUTE DECISION
# ==================================================

@dataclass
class RouteDecision:
    """
    Structured decision made by the LLM router.
    """

    # Main routing information
    tool: str
    answerable: bool = True
    reason: Optional[str] = None
    operation: Optional[str] = None

    # Analysis fields
    group_column: Optional[str] = None
    value_column: Optional[str] = None
    filter_column: Optional[str] = None
    filter_value: Optional[str] = None

    # Visualization fields
    chart_type: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None

    # Date/time fields
    date_column: Optional[str] = None
    time_granularity: Optional[str] = None
    filter_year: Optional[int] = None


# ==================================================
# ROUTER
# ==================================================

def route_question(
    question: str,
    df: Optional[pd.DataFrame] = None,
    knowledge_base_available: bool = False,
) -> RouteDecision:
    """
    Analyze a user's question and determine which
    tool should handle it.

    Supported sources:

    1. Structured dataset
    2. Document knowledge base

    Supported tools:

    - SQL
    - Python
    - Visualization
    - RAG
    - None
    """

    # ==================================================
    # VALIDATE QUESTION
    # ==================================================

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )


    # ==================================================
    # DETECT AVAILABLE SOURCES
    # ==================================================

    dataset_available = (
        df is not None
        and isinstance(df, pd.DataFrame)
        and not df.empty
    )

    if (
        not dataset_available
        and not knowledge_base_available
    ):

        raise ValueError(
            "No dataset or knowledge base is available."
        )


    # ==================================================
    # DATASET INFORMATION
    # ==================================================

    if dataset_available:

        dataset_schema = get_dataset_schema(
            df
        )

        dataset_columns = (
            df.columns.tolist()
        )

        numerical_columns = (
            df.select_dtypes(
                include="number"
            )
            .columns
            .tolist()
        )

        categorical_columns = (
            df.select_dtypes(
                include=["object", "string"]
            )
            .columns
            .tolist()
        )

        date_columns = (
            df.select_dtypes(
                include=["datetime", "datetimetz"]
            )
            .columns
            .tolist()
        )

        dataset_information = f"""
AVAILABLE DATASET COLUMNS
-------------------------
{dataset_columns}

NUMERICAL COLUMNS
-----------------
{numerical_columns}

CATEGORICAL COLUMNS
-------------------
{categorical_columns}

DATE/TIME COLUMNS
-----------------
{date_columns}

DATASET SCHEMA
--------------
{dataset_schema}
"""

    else:

        dataset_information = """
NO STRUCTURED DATASET IS CURRENTLY AVAILABLE.

The user has not uploaded a dataset.

Therefore:

- Do NOT select SQL.
- Do NOT select Python analysis.
- Do NOT select Visualization requiring dataset data.
- Use RAG if the uploaded documents can answer the question.
- Otherwise select none.
"""


    # ==================================================
    # KNOWLEDGE BASE INFORMATION
    # ==================================================

    if knowledge_base_available:

        knowledge_base_status = """
AVAILABLE

Indexed documents are available for RAG.

The documents may contain:

- policies
- procedures
- guidelines
- manuals
- reports
- business rules
- technical documentation
- regulations
- other textual business knowledge
"""

    else:

        knowledge_base_status = """
NOT AVAILABLE

There are currently no indexed documents available
for RAG.
"""


    # ==================================================
    # ROUTER PROMPT
    # ==================================================

    prompt = f"""
You are the routing system for a generic AI Data Analyst.

Your task is to analyze the user's question and return
ONE structured JSON routing decision.

The application can have TWO information sources:

1. STRUCTURED DATASET
2. UNSTRUCTURED DOCUMENT KNOWLEDGE BASE

You must determine which source can answer the question
and which tool should be used.


==================================================
USER QUESTION
==================================================

{question}


==================================================
STRUCTURED DATASET
==================================================

{dataset_information}


==================================================
DOCUMENT KNOWLEDGE BASE
==================================================

{knowledge_base_status}


==================================================
SOURCE AVAILABILITY
==================================================

Structured Dataset Available:
{dataset_available}

Document Knowledge Base Available:
{knowledge_base_available}


==================================================
CRITICAL DATASET RULE
==================================================

The dataset information above is the ONLY source of
truth for structured data.

Follow these rules:

1. Use only columns that actually exist.

2. Use column names EXACTLY as provided.

3. Preserve column capitalization.

4. NEVER invent columns.

5. NEVER assume common columns such as:

   sales
   profit
   revenue
   region
   category
   date
   customer
   salary

6. If the requested information does not exist in the
   dataset, do NOT force SQL, Python, or Visualization.

7. A question that cannot be answered by the dataset
   may still be answerable by the document knowledge base.


==================================================
TWO-SOURCE ANSWERABILITY
==================================================

Determine whether the question can be answered using
ANY currently available source.

SOURCE 1 — DATASET

Use the dataset for:

- numerical calculations
- aggregations
- filtering
- grouping
- rankings
- statistics
- structured records
- dataset-specific visualizations


SOURCE 2 — DOCUMENT KNOWLEDGE BASE

Use the knowledge base for information contained
in uploaded documents.

Examples:

- policies
- procedures
- guidelines
- manuals
- regulations
- business rules
- reports
- documentation
- textual explanations


IMPORTANT:

If the dataset cannot answer the question, DO NOT
immediately select none.

First determine whether the document knowledge base
can answer it.


==================================================
RAG RULE
==================================================

If:

Document Knowledge Base Available = true

RAG may be selected.

If:

Document Knowledge Base Available = false

RAG MUST NOT be selected.


==================================================
RAG EXAMPLES
==================================================

Question:

"What does the hiring process look like?"

If relevant uploaded documents are available:

tool = "rag"
answerable = true


Question:

"What does the uploaded policy say about discounts?"

If relevant uploaded documents are available:

tool = "rag"
answerable = true


Question:

"When is a region considered high-performing?"

If a relevant business document contains this
information:

tool = "rag"
answerable = true


Question:

"What is the company's employee policy?"

If no documents are available:

tool = "none"
answerable = false

reason should explain that no document knowledge
base is available.


==================================================
DATASET EXAMPLES
==================================================

Dataset:

region
sales
profit

Question:

"What is the total sales by region?"

tool = "sql"
answerable = true


Question:

"What is the average sales?"

tool = "python"
answerable = true


Question:

"Show a bar chart of sales by region."

tool = "visualization"
answerable = true


==================================================
COUNT RULE
==================================================

Do NOT interpret every question containing
the word "count" as COUNT(*).

Example:

"How many records are there?"

means:

count dataset rows.

Therefore:

tool = "sql"
operation = "count"


But:

"How many employees are there?"

requires employee information.

If employee information does not exist in the
dataset, do NOT count dataset rows.

First check whether documents can answer it.

If neither source can answer it:

tool = "none"
answerable = false


==================================================
AVAILABLE TOOLS
==================================================

1. SQL
-------

Use SQL for structured database questions such as:

- totals
- rankings
- top records
- bottom records
- filtering
- counting
- grouped queries
- aggregations
- selecting records
- highest values
- lowest values


2. PYTHON
---------

Use Python for analytical/statistical operations:

- mean
- average
- median
- standard deviation
- sum
- minimum
- maximum
- count
- correlation
- grouped analysis
- filtered analysis


3. VISUALIZATION
----------------

Use visualization when the user explicitly requests:

- chart
- graph
- plot
- visualization
- bar chart
- line chart
- pie chart
- scatter plot


4. RAG
------

Use RAG when the answer should come from uploaded
unstructured documents.


5. NONE
-------

Use none when no available source can answer
the question.


==================================================
ROUTING PRIORITY
==================================================

RULE 1 — EXPLICIT VISUALIZATION
--------------------------------

If the user explicitly asks for:

- chart
- graph
- plot
- visualization
- bar chart
- line chart
- pie chart
- scatter plot

use:

tool = "visualization"

ONLY when the required data exists in the dataset.

If the required chart data does not exist:

tool = "none"

UNLESS the requested information can be retrieved
from documents in a way that supports the requested
visualization.


RULE 2 — DOCUMENT KNOWLEDGE
----------------------------

If the question requires information from uploaded
documents AND the knowledge base is available:

tool = "rag"
answerable = true


RULE 3 — STATISTICAL ANALYSIS
-----------------------------

If the question requires:

- average
- mean
- median
- standard deviation
- correlation

and the required data exists:

tool = "python"


RULE 4 — STRUCTURED DATA
------------------------

For structured questions such as:

- total
- ranking
- filtering
- counting
- highest
- lowest
- grouped database queries

use:

tool = "sql"


RULE 5 — NOT ANSWERABLE
-----------------------

If neither available source can answer the question:

tool = "none"
answerable = false

Provide a short explanation in:

reason


==================================================
PYTHON OPERATIONS
==================================================

Map user language to operations:

average
→ mean

average value
→ mean

mean
→ mean

median
→ median

standard deviation
→ std

std deviation
→ std

standard dev
→ std

total
→ sum

sum
→ sum

minimum
→ min

lowest
→ min

maximum
→ max

highest
→ max

count
→ count

number of records
→ count


==================================================
GROUPED ANALYSIS
==================================================

When the question asks for analysis by another
column, identify:

group_column
value_column
operation

Example:

Dataset:

department
salary

Question:

"average salary by department"

return:

tool = "python"
operation = "mean"
group_column = "department"
value_column = "salary"


==================================================
FILTERED ANALYSIS
==================================================

When the question refers to a specific value,
identify:

filter_column
filter_value
value_column
operation

Example:

Dataset:

department
salary

Question:

"average salary in the IT department"

return:

tool = "python"
operation = "mean"
filter_column = "department"
filter_value = "IT"
value_column = "salary"


==================================================
VISUALIZATION
==================================================

BAR CHART
---------

Use bar charts for:

- categorical comparisons
- rankings
- grouped values


PIE CHART
---------

Use pie charts for:

- proportions
- shares
- part-to-whole comparisons


LINE CHART
----------

Use line charts for:

- trends over time
- daily values
- monthly values
- yearly values


SCATTER PLOT
------------

Use scatter plots for relationships between
two numerical variables.

Both x_column and y_column MUST exist.


==================================================
DATE/TIME RULES
==================================================

Do NOT assume columns such as:

date
order_date
month
year
quarter

Look at the actual DATE/TIME COLUMNS.

If the user asks for time-based analysis:

date_column = actual dataset date/time column

time_granularity must be one of:

day
month
year

If the user specifies a year such as 2025:

filter_year = 2025

Do NOT invent date-related columns.


==================================================
OUTPUT RULES
==================================================

Return ONLY valid JSON.

Do NOT return:

- markdown
- explanations outside JSON
- comments
- additional text

If a field is not required:

return null.

If:

answerable = false

then:

tool = "none"


==================================================
JSON FORMAT
==================================================

{{
    "tool": "sql | python | visualization | rag | none",

    "answerable": true,

    "reason": "explanation | null",

    "operation":
        "mean | median | std | sum | min | max | count | null",

    "group_column":
        "actual dataset column | null",

    "value_column":
        "actual dataset column | null",

    "filter_column":
        "actual dataset column | null",

    "filter_value":
        "value | null",

    "chart_type":
        "bar | line | pie | scatter | null",

    "x_column":
        "actual numerical dataset column | null",

    "y_column":
        "actual numerical dataset column | null",

    "date_column":
        "actual date/time dataset column | null",

    "time_granularity":
        "day | month | year | null",

    "filter_year":
        "integer year | null"
}}
"""


    # ==================================================
    # CALL GEMINI
    # ==================================================

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )

    response_text = response.text.strip()


    # ==================================================
    # REMOVE MARKDOWN CODE BLOCK
    # ==================================================

    if response_text.startswith("```"):

        response_text = (
            response_text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )


    # ==================================================
    # PARSE JSON
    # ==================================================

    try:

        data = json.loads(
            response_text
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            "Gemini returned invalid JSON:\n"
            f"{response_text}"
        ) from error


    # ==================================================
    # NORMALIZE TOOL
    # ==================================================

    tool = data.get(
        "tool"
    )

    if isinstance(tool, str):

        tool = tool.lower().strip()

        data["tool"] = tool


    # ==================================================
    # NORMALIZE ANSWERABILITY
    # ==================================================

    answerable = data.get(
        "answerable",
        True,
    )

    if isinstance(
        answerable,
        str,
    ):

        answerable = (
            answerable.lower().strip()
            in {
                "true",
                "yes",
                "1",
            }
        )


    # ==================================================
    # DETERMINISTIC OPERATION FIX
    # ==================================================

    question_lower = (
        question.lower()
    )

    if (
        "standard deviation"
        in question_lower
        or "std deviation"
        in question_lower
        or "standard dev"
        in question_lower
    ):

        data["operation"] = "std"


    # ==================================================
    # VALIDATE RAG AVAILABILITY
    # ==================================================

    if data.get("tool") == "rag":

        if not knowledge_base_available:

            answerable = False

            data["tool"] = "none"

            data["reason"] = (
                "No document knowledge base is "
                "currently available to answer "
                "this question."
            )


    # ==================================================
    # VALIDATE DATASET-BASED TOOLS
    # ==================================================

    dataset_tools = {
        "sql",
        "python",
        "visualization",
    }

    if (
        data.get("tool") in dataset_tools
        and not dataset_available
    ):

        answerable = False

        data["tool"] = "none"

        data["reason"] = (
            "This question requires structured "
            "dataset information, but no dataset "
            "is currently available."
        )


    # ==================================================
    # HANDLE NOT ANSWERABLE
    # ==================================================

    if not answerable:

        data["tool"] = "none"

        data["operation"] = None
        data["group_column"] = None
        data["value_column"] = None
        data["filter_column"] = None
        data["filter_value"] = None

        data["chart_type"] = None
        data["x_column"] = None
        data["y_column"] = None

        data["date_column"] = None
        data["time_granularity"] = None
        data["filter_year"] = None

        if not data.get("reason"):

            data["reason"] = (
                "The question cannot be answered "
                "using the available data sources."
            )


    # ==================================================
    # RETURN ROUTE DECISION
    # ==================================================

    return RouteDecision(
        tool=data.get(
            "tool",
            "none",
        ),

        answerable=answerable,

        reason=data.get(
            "reason"
        ),

        operation=data.get(
            "operation"
        ),

        group_column=data.get(
            "group_column"
        ),

        value_column=data.get(
            "value_column"
        ),

        filter_column=data.get(
            "filter_column"
        ),

        filter_value=data.get(
            "filter_value"
        ),

        chart_type=data.get(
            "chart_type"
        ),

        x_column=data.get(
            "x_column"
        ),

        y_column=data.get(
            "y_column"
        ),

        date_column=data.get(
            "date_column"
        ),

        time_granularity=data.get(
            "time_granularity"
        ),

        filter_year=data.get(
            "filter_year"
        ),
    )