from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.analysis.analyzer import group_by_analysis
from ai_data_analyst.visualization.charts import create_bar_chart


# Load dataset
df = load_csv("data/raw/sales.csv")


# Calculate total sales by region
result = group_by_analysis(
    df=df,
    group_column="region",
    value_column="sales",
    operation="sum"
)


print("\n========== CHART DATA ==========\n")
print(result)


# Create chart
fig = create_bar_chart(
    df=result,
    x_column="region",
    y_column="sum_sales",
    title="Total Sales by Region"
)


# Save chart
output_path = "data/processed/sales_by_region.png"

fig.savefig(output_path)

print(f"\nChart saved successfully to: {output_path}")