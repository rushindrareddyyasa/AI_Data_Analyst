from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.analysis.analyzer import filtered_aggregation


df = load_csv("data/raw/sales.csv")


result = filtered_aggregation(
    df=df,
    filter_column="region",
    filter_value="West",
    value_column="sales",
    operation="mean"
)

print("\n========== AVERAGE SALES: WEST ==========\n")
print(f"Average sales: {result:.2f}")