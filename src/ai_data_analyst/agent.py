import pandas as pd

from ai_data_analyst.llm.router import route_question
from ai_data_analyst.llm.validator import validate_route

from ai_data_analyst.sql.query_engine import answer_question

from ai_data_analyst.analysis.executor import execute_analysis

from ai_data_analyst.visualization.executor import (
    execute_visualization,
)

from ai_data_analyst.llm.response import (
    explain_result,
    explain_scalar_result,
)

from ai_data_analyst.rag.embeddings import EmbeddingModel
from ai_data_analyst.rag.vector_store import VectorStore
from ai_data_analyst.rag.retriever import Retriever
from ai_data_analyst.rag.pipeline import RAGPipeline
from ai_data_analyst.llm.client import client


def run_agent(
    df,
    engine,
    question: str,
):
    """
    Main AI Data Analyst agent.

    The agent:
    1. Routes the user question.
    2. Validates the routing decision.
    3. Executes the selected tool.
    4. Returns the final answer and relevant output.
    """

    # --------------------------------
    # ROUTING
    # --------------------------------

    decision = route_question(
        question=question,
        df=df,
    )
    if not decision.answerable:
        return {
            "tool": "none",
            "decision": decision,
            "answer": (
                decision.reason
                or "I cannot answer this question "
                "using the uploaded dataset."
            ),
            "result": None,
            "figure": None,
            "sources": [],
        }

    print(
        "\n========== ROUTING DECISION ==========\n"
    )

    print(decision)

    # --------------------------------
    # ROUTE VALIDATION
    # --------------------------------

    validate_route(
        decision,
        df,
    )

    print(
        "\n========== ROUTE VALIDATION ==========\n"
    )

    print("Route validated successfully.")

    # --------------------------------
    # SQL
    # --------------------------------

    if decision.tool == "sql":

        answer = answer_question(
            df=df,
            engine=engine,
            question=question,
        )

        return {
            "tool": "sql",
            "decision": decision,
            "answer": answer,
            "result": None,
            "figure": None,
            "sources": [],
        }

    # --------------------------------
    # PYTHON
    # --------------------------------

    if decision.tool == "python":

        result = execute_analysis(
            df,
            decision,
        )

        if not isinstance(
            result,
            pd.DataFrame,
        ):

            answer = explain_scalar_result(
                question=question,
                result=result,
                operation=decision.operation,
                value_column=decision.value_column,
            )

        else:

            answer = explain_result(
                question,
                result,
            )

        return {
            "tool": "python",
            "decision": decision,
            "answer": answer,
            "result": result,
            "figure": None,
            "sources": [],
        }

    # --------------------------------
    # VISUALIZATION
    # --------------------------------

    if decision.tool == "visualization":

        figure = execute_visualization(
            df,
            decision,
        )

        return {
            "tool": "visualization",
            "decision": decision,
            "answer": (
                f"Created a "
                f"{decision.chart_type} chart."
            ),
            "result": None,
            "figure": figure,
            "sources": [],
        }

    # --------------------------------
    # RAG
    # --------------------------------

    if decision.tool == "rag":

        embedding_model = EmbeddingModel(
            client
        )

        vector_store = VectorStore()

        retriever = Retriever(
            embedding_model,
            vector_store,
        )

        rag_pipeline = RAGPipeline(
            retriever
        )

        rag_result = (
            rag_pipeline.answer_with_sources(
                question=question,
                top_k=3,
            )
        )

        return {
            "tool": "rag",
            "decision": decision,
            "answer": rag_result["answer"],
            "result": None,
            "figure": None,
            "sources": rag_result["sources"],
        }

    # --------------------------------
    # UNKNOWN TOOL
    # --------------------------------

    raise ValueError(
        f"Unsupported tool: {decision.tool}"
    )