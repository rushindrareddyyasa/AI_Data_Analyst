from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.sql.database import create_database


# Load dataset
df = load_csv("data/raw/sales.csv")

# Create database
engine = create_database(df)

print("Database created successfully!")

print("\nDatabase URL:")
print(engine.url)