from ai_data_analyst.ingestion.loader import load_csv


df = load_csv("data/raw/sales.csv")

print(df.head())
print(df.shape)
print(df.columns.tolist())