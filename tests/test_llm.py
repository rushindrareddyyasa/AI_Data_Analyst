from ai_data_analyst.llm.client import generate_response


prompt = """
You are an AI Data Analyst.

Explain what a sales dataset is in 3 simple sentences.
"""


response = generate_response(prompt)

print("\n========== GEMINI RESPONSE ==========\n")
print(response)