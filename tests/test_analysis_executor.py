from ai_data_analyst.ingestion.loader import load_csv
from ai_data_analyst.llm.router import route_question
from ai_data_analyst.analysis.executor import execute_analysis


df = load_csv("data/raw/sales.csv")


questions = [
    "What is the average profit?",
    "What is the median profit?",
    "What is the standard deviation of profit?",
    "What is the average profit by category?",
    "What is the average sales in the West region?",
]


for question in questions:

    print("\n========================================")
    print(f"QUESTION: {question}")

    decision = route_question(question,df)

    print("\nROUTING DECISION:")
    print(decision)

    result = execute_analysis(
        df,
        decision
    )

    print("\nANALYSIS RESULT:")
    print(result)