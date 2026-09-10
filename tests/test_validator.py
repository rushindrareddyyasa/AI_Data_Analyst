from ai_data_analyst.sql.validator import validate_sql


safe_queries = [
    "SELECT region FROM sales;",
    "SELECT SUM(sales) FROM sales;",
    "SELECT region, SUM(sales) FROM sales GROUP BY region;",
]

dangerous_queries = [
    "DELETE FROM sales;",
    "DROP TABLE sales;",
    "UPDATE sales SET profit = 0;",
    "INSERT INTO sales VALUES (1);",
]


print("========== SAFE QUERIES ==========")

for query in safe_queries:

    result = validate_sql(query)

    print(f"{result} | {query}")


print("\n========== DANGEROUS QUERIES ==========")

for query in dangerous_queries:

    result = validate_sql(query)

    print(f"{result} | {query}")