from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.analysis.analyzer import calculate_summary


# Load dataset
df = load_csv("data/raw/sales.csv")

# Calculate numerical summary
summary = calculate_summary(df)

print("\n========== NUMERICAL SUMMARY ==========\n")

for statistic, values in summary.items():
    print(f"\n--- {statistic.upper()} ---")

    for column, value in values.items():
        print(f"{column}: {value:.2f}")