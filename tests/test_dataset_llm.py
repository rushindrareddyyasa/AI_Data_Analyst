from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.ingestion.profiler import profile_dataset
from ai_data_analyst.llm.client import generate_response
from ai_data_analyst.llm.prompts import create_dataset_prompt


# Load dataset
df = load_csv("data/raw/sales.csv")

# Profile dataset
profile = profile_dataset(df)

# User question
question = "Give me an overview of this dataset."

# Create prompt
prompt = create_dataset_prompt(
    profile,
    question
)

# Ask Gemini
response = generate_response(prompt)

print("\n========== AI DATA ANALYST ==========\n")
print(response)