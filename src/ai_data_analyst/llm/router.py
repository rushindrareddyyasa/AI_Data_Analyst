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
    df: pd.DataFrame,
) -> RouteDecision:
    """
    Analyze a user's question and determine which
    tool should handle it.

    The router is dataset-aware and uses the actual
    uploaded dataset schema as the source of truth.
    """

    # --------------------------------------------------
    # Validate question
    # --------------------------------------------------

    if not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if df.empty:
        raise ValueError(
            "Dataset cannot be empty."
        )

    # --------------------------------------------------
    # Generate dynamic dataset information
    # --------------------------------------------------

    dataset_schema = get_dataset_schema(df)

    dataset_columns = df.columns.tolist()

    numerical_columns = (
        df.select_dtypes(include="number")
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
            include=["datetime"]
        )
        .columns
        .tolist()
    )

    # --------------------------------------------------
    # Create dataset metadata for LLM
    # --------------------------------------------------

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

    # ==================================================
    # ROUTER PROMPT
    # ==================================================

    prompt = f"""
You are the routing system for a generic AI Data Analyst.

Your job is to analyze the user's question and return
ONE structured JSON routing decision.

You are NOT restricted to a sales dataset.

The user may upload any structured dataset such as:

- Sales
- Finance
- Employees
- Students
- Customers
- Healthcare
- Marketing
- Operations
- Manufacturing
- Logistics
- Any other tabular dataset


==================================================
USER QUESTION
==================================================

{question}


==================================================
ACTUAL DATASET INFORMATION
==================================================

{dataset_information}


==================================================
CRITICAL DATASET RULE
==================================================

The ACTUAL DATASET INFORMATION above is the
ONLY source of truth for dataset columns.

Follow these rules:

1. Use only columns that exist in the dataset.

2. Use column names EXACTLY as they appear.

3. Preserve column capitalization.

4. NEVER invent a column.

5. NEVER assume common columns such as:
   sales, profit, revenue, region, category,
   date, customer, salary, etc.

6. If the dataset contains:

   SALES

   and the user asks:

   "What is the total sales?"

   use:

   value_column = "SALES"

7. If the dataset contains:

   sales

   then use:

   value_column = "sales"

8. If the requested concept does not have a suitable
   column, metric, or information in the dataset,
   mark the question as NOT ANSWERABLE.


==================================================
QUESTION-DATASET RELEVANCE
==================================================

Before selecting a tool, determine whether the
user's question can actually be answered using
the available dataset.

Set:

answerable = true

when the dataset contains the information required
to answer the question.

Set:

answerable = false

when the question asks for information that is not
represented in the dataset.

Examples:

Dataset:

Destination
State
Category
Average Budget

Question:

"What is the average budget?"

answerable = true


Dataset:

Destination
State
Category
Average Budget

Question:

"What is the employee count?"

answerable = false

reason:

"The dataset does not contain employee information."


Dataset:

Destination
State
Category
Average Budget

Question:

"What is the average employee salary?"

answerable = false

reason:

"The dataset does not contain employee salary information."


==================================================
COUNT RULE
==================================================

Do NOT interpret every word containing "count"
as COUNT(*).

There is an important difference between:

"How many records are there?"

and:

"How many employees are there?"

"How many records are there?"

means:

count the rows in the dataset.

Therefore:

answerable = true
operation = count


But:

"How many employees are there?"

requires employee-related information.

If employee information does not exist:

answerable = false


Similarly:

"How many customers are there?"

should only be answered as a customer count
if the dataset contains customer-related information.

Do not blindly return the total number of rows.


==================================================
ANSWERABILITY RULES
==================================================

Set answerable = false when:

- the requested entity does not exist in the dataset
- the requested metric does not exist and cannot
  reasonably be derived from available columns
- the question requires information outside the dataset
- the question refers to a different business domain
- answering would require inventing a column

When answerable = false:

1. Do not invent a column.
2. Do not select a fabricated column.
3. Do not select SQL, Python, or visualization
   for executing the question.
4. Set tool = "none".
5. Set operation = null.
6. Set relevant analysis fields to null.
7. Provide a short explanation in reason.


==================================================
AVAILABLE TOOLS
==================================================

1. SQL
-------

Use SQL for structured database questions such as:

- totals
- rankings
- top/bottom records
- filtering
- counting
- grouped database queries
- aggregations
- selecting records
- finding highest/lowest values


2. PYTHON
---------

Use Python for statistical and analytical operations such as:

- mean
- median
- standard deviation
- sum
- minimum
- maximum
- count
- correlation
- statistical analysis
- grouped analysis
- filtered analysis


3. VISUALIZATION
----------------

Use visualization when the user explicitly asks for:

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

Use RAG when the question requires information
from unstructured documents.

Examples:

- company policies
- business rules
- guidelines
- procedures
- documentation
- regulations
- internal instructions
- text-based knowledge

RAG is NOT for numerical analysis of the uploaded
structured dataset.


==================================================
ROUTING PRIORITY
==================================================

Follow these rules in order.

RULE 0
------

First determine whether the question is answerable
from the dataset.

If not:

tool = "none"
answerable = false


RULE 1
------

If the user explicitly requests a:

- chart
- graph
- plot
- visualization

then:

tool = "visualization"


RULE 2
------

If the user asks about:

- policies
- documents
- procedures
- guidelines
- company rules
- internal documentation

then:

tool = "rag"


RULE 3
------

If the question requires statistical analysis such as:

- average
- median
- standard deviation
- correlation

then:

tool = "python"


RULE 4
------

For ordinary structured-data questions such as:

- total
- ranking
- filtering
- counting
- highest
- lowest
- grouped database queries

use:

tool = "sql"


==================================================
PYTHON OPERATIONS
==================================================

Map user language to operations:

average / average value
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

minimum / lowest
→ min

maximum / highest
→ max

count / number of records
→ count


==================================================
GROUPED ANALYSIS
==================================================

When the question asks for analysis by another
column, identify:

group_column
value_column
operation

For example, if the actual dataset contains:

department
salary

and the user asks:

"average salary by department"

return:

tool = "python"
operation = "mean"
group_column = "department"
value_column = "salary"

IMPORTANT:

Use the actual dataset column names.


==================================================
FILTERED ANALYSIS
==================================================

When the question refers to a specific value,
identify:

filter_column
filter_value
value_column
operation

For example, if the dataset contains:

department
salary

and the user asks:

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

Use scatter plots for:

- relationships between two numerical variables


For scatter plots:

x_column = first numerical variable
y_column = second numerical variable

Both columns MUST exist in the dataset.


==================================================
DATE/TIME RULES
==================================================

Do NOT assume a column called:

order_date
date
month
year
quarter

Instead:

1. Look at DATE/TIME COLUMNS from the actual dataset.

2. If the user asks for time-based analysis,
   select an actual date/time column.

3. Use:

day
month
year

for time_granularity.

4. If the user specifies a year such as 2025,
   set:

filter_year = 2025

5. Do NOT create fake columns such as:

month
year
quarter

in group_column or filter_column.

6. Use date_column for the actual date column.


==================================================
CHART ROUTING EXAMPLES
==================================================

These examples are illustrative only.

They do NOT define the available dataset columns.

Example:

User:
"Show the average value by department as a bar chart."

If the actual dataset contains:

department
salary

then:

tool = "visualization"
operation = "mean"
group_column = "department"
value_column = "salary"
chart_type = "bar"


Example:

User:
"Show the relationship between age and salary."

If the actual dataset contains:

age
salary

then:

tool = "visualization"
chart_type = "scatter"
x_column = "age"
y_column = "salary"


Example:

User:
"Show the monthly revenue trend."

If the actual dataset contains:

revenue
transaction_date

then:

tool = "visualization"
operation = "sum"
value_column = "revenue"
chart_type = "line"
date_column = "transaction_date"
time_granularity = "month"


==================================================
RAG ROUTING
==================================================

If the question requires information contained
in company documents:

tool = "rag"

Examples:

"When is a region considered high-performing?"

"What is the company discount policy?"

"What does the employee handbook say about leave?"

"What are the internal reporting guidelines?"


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

If the question is not answerable:

tool must be "none"
answerable must be false
reason must explain why.


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

        data = json.loads(response_text)

    except json.JSONDecodeError as error:

        raise ValueError(
            "Gemini returned invalid JSON:\n"
            f"{response_text}"
        ) from error

    # ==================================================
    # DETERMINISTIC FALLBACKS
    # ==================================================

    question_lower = question.lower()

    if (
        "standard deviation" in question_lower
        or "std deviation" in question_lower
        or "standard dev" in question_lower
    ):

        data["operation"] = "std"

    # ==================================================
    # NORMALIZE ANSWERABILITY
    # ==================================================

    answerable = data.get(
        "answerable",
        True,
    )

    # Gemini may occasionally return strings
    # instead of a real boolean.

    if isinstance(answerable, str):

        answerable = (
            answerable.lower()
            in {"true", "yes", "1"}
        )

    # --------------------------------------------------
    # If not answerable, force tool to none
    # --------------------------------------------------

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
                "using the uploaded dataset."
            )

    # ==================================================
    # RETURN ROUTE DECISION
    # ==================================================

    return RouteDecision(
        tool=data.get("tool"),
        answerable=answerable,
        reason=data.get("reason"),
        operation=data.get("operation"),
        group_column=data.get("group_column"),
        value_column=data.get("value_column"),
        filter_column=data.get("filter_column"),
        filter_value=data.get("filter_value"),
        chart_type=data.get("chart_type"),
        x_column=data.get("x_column"),
        y_column=data.get("y_column"),
        date_column=data.get("date_column"),
        time_granularity=data.get("time_granularity"),
        filter_year=data.get("filter_year"),
    )