import pandas as pd

from ai_data_analyst.llm.router import route_question
from ai_data_analyst.llm.validator import validate_route

from ai_data_analyst.sql.query_engine import answer_question

from ai_data_analyst.analysis.executor import (
    execute_analysis,
    generate_analysis_code,
)

from ai_data_analyst.visualization.executor import (
    execute_visualization,
    generate_visualization_code,
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


# ==================================================
# AGENT
# ==================================================

def run_agent(
    df,
    engine,
    question: str,
):
    """
    Main AI agent responsible for routing a user's
    natural-language question to the appropriate tool.

    Supported tools:

    - SQL
    - Python
    - Visualization
    - RAG
    - None

    Dataset is optional when the question can be
    answered entirely from the document knowledge base.
    """

    # ==================================================
    # VALIDATE QUESTION
    # ==================================================

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )


    # ==================================================
    # DETECT DATASET AVAILABILITY
    # ==================================================

    dataset_available = (
        df is not None
        and isinstance(df, pd.DataFrame)
        and not df.empty
    )


    # ==================================================
    # DETECT KNOWLEDGE BASE AVAILABILITY
    # ==================================================

    vector_store = VectorStore()

    knowledge_base_available = (
        vector_store.count() > 0
    )


    # ==================================================
    # VALIDATE INPUT SOURCES
    # ==================================================

    if not dataset_available and not knowledge_base_available:

        return {
            "tool": "none",
            "decision": None,
            "answer": (
                "Please upload a dataset or at least "
                "one document before asking a question."
            ),
            "result": None,
            "sql": None,
            "code": None,
            "figure": None,
            "sources": [],
        }


    # ==================================================
    # ROUTE QUESTION
    # ==================================================

    decision = route_question(
        question=question,
        df=df,
        knowledge_base_available=knowledge_base_available,
    )


    # ==================================================
    # HANDLE UNANSWERABLE QUESTIONS
    # ==================================================

    if not decision.answerable:

        return {
            "tool": "none",
            "decision": decision,
            "answer": (
                decision.reason
                or "The question cannot be answered "
                "using the available information."
            ),
            "result": None,
            "sql": None,
            "code": None,
            "figure": None,
            "sources": [],
        }


    # ==================================================
    # VALIDATE ROUTE
    # ==================================================

    # SQL / Python / Visualization require a dataset.
    # RAG only requires the document knowledge base.

    if decision.tool != "rag":

        if not dataset_available:

            return {
                "tool": "none",
                "decision": decision,
                "answer": (
                    "This question requires a dataset, "
                    "but no dataset has been uploaded."
                ),
                "result": None,
                "sql": None,
                "code": None,
                "figure": None,
                "sources": [],
            }

        validate_route(
            decision,
            df,
        )


    # ==================================================
    # SQL
    # ==================================================

    if decision.tool == "sql":

        sql_result = answer_question(
            df=df,
            engine=engine,
            question=question,
        )

        return {
            "tool": "sql",
            "decision": decision,
            "answer": sql_result["answer"],
            "result": sql_result["result"],
            "sql": sql_result["sql"],
            "code": None,
            "figure": None,
            "sources": [],
        }


    # ==================================================
    # PYTHON ANALYSIS
    # ==================================================

    if decision.tool == "python":

        code = generate_analysis_code(
            decision
        )

        result = execute_analysis(
            df,
            decision,
        )

        if isinstance(
            result,
            pd.DataFrame,
        ):

            answer = explain_result(
                question=question,
                result=result,
            )

        else:

            answer = explain_scalar_result(
                question=question,
                result=result,
                operation=decision.operation,
                value_column=decision.value_column,
            )

        return {
            "tool": "python",
            "decision": decision,
            "answer": answer,
            "result": result,
            "sql": None,
            "code": code,
            "figure": None,
            "sources": [],
        }


    # ==================================================
    # VISUALIZATION
    # ==================================================

    if decision.tool == "visualization":

        code = generate_visualization_code(
            decision
        )

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
            "sql": None,
            "code": code,
            "figure": figure,
            "sources": [],
        }


    # ==================================================
    # RAG
    # ==================================================

    if decision.tool == "rag":

        if not knowledge_base_available:

            return {
                "tool": "none",
                "decision": decision,
                "answer": (
                    "No documents are currently "
                    "available in the knowledge base."
                ),
                "result": None,
                "sql": None,
                "code": None,
                "figure": None,
                "sources": [],
            }


        # ----------------------------------------------
        # Embedding Model
        # ----------------------------------------------

        embedding_model = EmbeddingModel(
            client
        )


        # ----------------------------------------------
        # Retriever
        # ----------------------------------------------

        retriever = Retriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
        )


        # ----------------------------------------------
        # RAG Pipeline
        # ----------------------------------------------

        rag_pipeline = RAGPipeline(
            retriever=retriever
        )


        # ----------------------------------------------
        # Generate Answer
        # ----------------------------------------------

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
            "sql": None,
            "code": None,
            "figure": None,
            "sources": rag_result["sources"],
        }


    # ==================================================
    # UNKNOWN TOOL
    # ==================================================

    raise ValueError(
        f"Unsupported tool selected by agent: "
        f"{decision.tool}"
    )


## Completed