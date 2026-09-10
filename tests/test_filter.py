from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.analysis.filter import filter_data


df = load_csv("data/raw/sales.csv")

result = filter_data(
    df,
    "region",
    "West"
)

print("\n========== FILTERED DATA ==========\n")
print(result.head())

print("\nOriginal rows:", len(df))
print("Filtered rows:", len(result))