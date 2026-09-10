import re


def validate_sql(sql: str) -> bool:
    """
    Validate LLM-generated SQL before execution.

    Only SELECT and WITH queries are allowed.
    """

    sql = sql.strip().lower()

    # Remove trailing semicolon
    sql = sql.rstrip(";").strip()

    # Only SELECT or WITH queries are allowed
    if not (sql.startswith("select") or sql.startswith("with")):
        return False

    # Block dangerous SQL operations
    forbidden_keywords = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "replace",
        "truncate",
        "attach",
        "detach",
        "pragma",
    ]

    for keyword in forbidden_keywords:
        pattern = rf"\b{keyword}\b"

        if re.search(pattern, sql):
            return False

    return True