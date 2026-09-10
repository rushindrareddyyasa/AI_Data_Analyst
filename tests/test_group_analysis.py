from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.analysis.analyzer import group_by_analysis


df = load_csv("data/raw/sales.csv")


print("\n========== SALES BY REGION ==========\n")

result = group_by_analysis(
    df=df,
    group_column="region",
    value_column="sales",
    operation="sum"
)

print(result)


print("\n========== AVERAGE PROFIT BY CATEGORY ==========\n")

result = group_by_analysis(
    df=df,
    group_column="category",
    value_column="profit",
    operation="mean"
)

print(result)