from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.visualization.charts import create_line_chart


df = load_csv("data/raw/sales.csv")

# Filter 2025
df_2025 = df[df["order_date"].dt.year == 2025].copy()

# Create month column
df_2025["month"] = df_2025["order_date"].dt.to_period("M").astype(str)

# Calculate monthly sales
monthly_sales = (
    df_2025
    .groupby("month")["sales"]
    .sum()
    .reset_index()
)

print("\n========== MONTHLY SALES 2025 ==========\n")
print(monthly_sales)


# Create chart
fig = create_line_chart(
    df=monthly_sales,
    x_column="month",
    y_column="sales",
    title="Monthly Sales — 2025"
)

# Save chart
output_path = "data/processed/monthly_sales_2025.png"
fig.savefig(output_path)

print(f"\nChart saved successfully to: {output_path}")