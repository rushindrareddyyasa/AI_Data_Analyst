from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.ingestion.profiler import profile_dataset


df = load_csv("data/raw/sales.csv")

profile = profile_dataset(df)


print("\n========== DATASET OVERVIEW ==========")

print(f"Rows: {profile['rows']}")
print(f"Columns: {profile['columns']}")
print(f"Duplicate Rows: {profile['duplicate_rows']}")


print("\n========== COLUMN TYPES ==========")

print("Numerical:")
print(profile["numerical_columns"])

print("\nCategorical:")
print(profile["categorical_columns"])

print("\nDate:")
print(profile["date_columns"])


print("\n========== COLUMN INFORMATION ==========")

for column, information in profile["column_information"].items():

    print(
        f"{column:20} | "
        f"type={information['data_type']:12} | "
        f"missing={information['missing_values']:3} | "
        f"missing%={information['missing_percentage']:5.2f}% | "
        f"unique={information['unique_values']}"
    )


print("\n========== NUMERICAL STATISTICS ==========")

for column, statistics in profile["numerical_statistics"].items():

    print(
        f"{column:20} | "
        f"min={statistics['min']:.2f} | "
        f"max={statistics['max']:.2f} | "
        f"mean={statistics['mean']:.2f} | "
        f"median={statistics['median']:.2f}"
    )


print("\n========== DATE RANGES ==========")

for column, date_range in profile["date_ranges"].items():

    print(
        f"{column}: "
        f"{date_range['min']} → {date_range['max']}"
    )