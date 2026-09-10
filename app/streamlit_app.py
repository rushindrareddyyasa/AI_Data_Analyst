import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))


# ==================================================
# PROJECT IMPORTS
# ==================================================

from ai_data_analyst.ingestion.dataset import load_dataset
from ai_data_analyst.sql.database import create_database
from ai_data_analyst.agent import run_agent
from ai_data_analyst.rag.ingest import ingest_uploaded_documents
from ai_data_analyst.rag.vector_store import VectorStore


# ==================================================
# STREAMLIT PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
)


# ==================================================
# SESSION STATE
# ==================================================

if "ingested_documents" not in st.session_state:
    st.session_state.ingested_documents = set()


# ==================================================
# APPLICATION HEADER
# ==================================================

st.title("🤖 AI Data Analyst")

st.write(
    "Upload a dataset, business documents, or both "
    "and interact with them using natural language."
)


# ==================================================
# SIDEBAR — DATASET
# ==================================================

st.sidebar.header("📊 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload a dataset",
    type=["csv", "xlsx"],
    help="Upload a CSV or Excel dataset.",
)


# ==================================================
# SIDEBAR — DOCUMENTS
# ==================================================

st.sidebar.header("📚 Documents")

uploaded_documents = st.sidebar.file_uploader(
    "Upload documents",
    type=["txt", "pdf"],
    accept_multiple_files=True,
    help=(
        "Upload TXT or PDF documents containing "
        "policies, guidelines, reports, or other "
        "business information."
    ),
)


# ==================================================
# APPLICATION VARIABLES
# ==================================================

df = None
engine = None
dataset_name = None


# ==================================================
# LOAD DATASET
# ==================================================

if uploaded_file is not None:

    try:

        df = load_dataset(
            uploaded_file
        )

        dataset_name = uploaded_file.name

        if df.empty:

            st.sidebar.error(
                "The uploaded dataset is empty."
            )

            df = None

        else:

            st.sidebar.success(
                f"Loaded: {dataset_name}"
            )

    except Exception as error:

        st.sidebar.error(
            f"Unable to load dataset: {error}"
        )

        df = None


# ==================================================
# CREATE DATABASE
# ==================================================

if df is not None:

    database_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "data.db"
    )

    try:

        engine = create_database(
            df=df,
            database_path=str(database_path),
            table_name="dataset",
        )

    except Exception as error:

        st.sidebar.error(
            f"Unable to create database: {error}"
        )

        engine = None


# ==================================================
# DOCUMENT INGESTION
# ==================================================

if uploaded_documents:

    new_documents = []

    for document in uploaded_documents:

        document_key = (
            f"{document.name}_{document.size}"
        )

        if (
            document_key
            not in st.session_state.ingested_documents
        ):

            new_documents.append(
                document
            )

    if new_documents:

        with st.sidebar:

            with st.spinner(
                "Processing documents..."
            ):

                try:

                    ingest_uploaded_documents(
                        uploaded_files=new_documents
                    )

                    for document in new_documents:

                        document_key = (
                            f"{document.name}_{document.size}"
                        )

                        st.session_state.ingested_documents.add(
                            document_key
                        )

                    st.success(
                        f"Processed "
                        f"{len(new_documents)} "
                        "document(s)."
                    )

                except Exception as error:

                    st.error(
                        f"Document ingestion failed: "
                        f"{error}"
                    )

    else:

        st.sidebar.info(
            "Uploaded documents are already indexed."
        )


# ==================================================
# KNOWLEDGE BASE STATUS
# ==================================================

try:

    vector_store = VectorStore()

    document_count = vector_store.count()

except Exception:

    document_count = 0


knowledge_base_available = (
    document_count > 0
)


# ==================================================
# SIDEBAR — DATASET INFORMATION
# ==================================================

if df is not None:

    with st.sidebar:

        st.divider()

        st.header(
            "📊 Dataset Information"
        )

        st.write(
            f"**File:** {dataset_name}"
        )

        st.write(
            f"**Rows:** {len(df):,}"
        )

        st.write(
            f"**Columns:** {len(df.columns)}"
        )

        st.divider()

        st.write("### Columns")

        for column in df.columns:

            st.write(
                f"- `{column}`"
            )


# ==================================================
# SIDEBAR — KNOWLEDGE BASE INFORMATION
# ==================================================

with st.sidebar:

    st.divider()

    st.header(
        "📚 Knowledge Base"
    )

    st.write(
        f"**Indexed chunks:** "
        f"{document_count}"
    )

    if uploaded_documents:

        st.write(
            "### Uploaded Documents"
        )

        for document in uploaded_documents:

            st.write(
                f"- `{document.name}`"
            )


# ==================================================
# SOURCE STATUS
# ==================================================

st.subheader(
    "🔌 Available Sources"
)

source_col1, source_col2 = st.columns(2)


with source_col1:

    if df is not None:

        st.success(
            f"📊 Dataset available — "
            f"{len(df):,} rows"
        )

    else:

        st.warning(
            "📊 No dataset uploaded"
        )


with source_col2:

    if knowledge_base_available:

        st.success(
            f"📚 Knowledge base available — "
            f"{document_count} indexed chunks"
        )

    else:

        st.warning(
            "📚 No documents indexed"
        )


# ==================================================
# DATASET PREVIEW
# ==================================================

if df is not None:

    st.subheader(
        "📋 Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
    )


# ==================================================
# QUESTION SECTION
# ==================================================

st.subheader(
    "💬 Ask Your Data or Documents"
)

question = st.text_input(
    "Ask a question",
    placeholder=(
        "Example: What is the average sales by region?"
    ),
)


# ==================================================
# ASK AI
# ==================================================

if st.button(
    "🚀 Ask AI",
    type="primary",
):

    # --------------------------------------------------
    # Validate Question
    # --------------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a question."
        )


    # --------------------------------------------------
    # Validate Sources
    # --------------------------------------------------

    elif (
        df is None
        and not knowledge_base_available
    ):

        st.warning(
            "Please upload a dataset or at least "
            "one document before asking a question."
        )


    # --------------------------------------------------
    # Run Agent
    # --------------------------------------------------

    else:

        with st.spinner(
            "Analyzing your question..."
        ):

            try:

                result = run_agent(
                    df=df,
                    engine=engine,
                    question=question,
                )


                # ==================================
                # ANSWER
                # ==================================

                st.subheader(
                    "💡 Answer"
                )

                st.write(
                    result["answer"]
                )


                # ==================================
                # ANALYSIS METHOD
                # ==================================

                st.subheader(
                    "🔧 Analysis Method"
                )

                tool_name = (
                    result["tool"]
                    .upper()
                )

                st.info(
                    tool_name
                )


                # ==================================
                # GENERATED LOGIC
                # ==================================

                generated_sql = result.get(
                    "sql"
                )

                generated_code = result.get(
                    "code"
                )

                if (
                    generated_sql
                    or generated_code
                ):

                    st.subheader(
                        "🧠 Generated Logic"
                    )

                    if generated_sql:

                        st.caption(
                            "SQL executed against "
                            "the uploaded dataset"
                        )

                        st.code(
                            generated_sql,
                            language="sql",
                        )

                    elif generated_code:

                        st.caption(
                            "Python logic executed "
                            "by the analysis engine"
                        )

                        st.code(
                            generated_code,
                            language="python",
                        )


                # ==================================
                # ANALYSIS RESULT
                # ==================================

                if result["result"] is not None:

                    st.subheader(
                        "📋 Analysis Result"
                    )

                    analysis_result = (
                        result["result"]
                    )


                    if isinstance(
                        analysis_result,
                        pd.DataFrame,
                    ):

                        st.dataframe(
                            analysis_result,
                            use_container_width=True,
                        )

                    else:

                        st.metric(
                            label="Calculated Value",
                            value=str(
                                analysis_result
                            ),
                        )


                # ==================================
                # VISUALIZATION
                # ==================================

                if result["figure"] is not None:

                    st.subheader(
                        "📈 Visualization"
                    )

                    st.pyplot(
                        result["figure"],
                        use_container_width=True,
                    )


                # ==================================
                # RAG SOURCES
                # ==================================

                if result["sources"]:

                    st.subheader(
                        "📚 Sources"
                    )

                    for source in result[
                        "sources"
                    ]:

                        st.write(
                            f"• `{source}`"
                        )


                # ==================================
                # AGENT DECISION DETAILS
                # ==================================

                decision = result.get(
                    "decision"
                )

                if decision is not None:

                    with st.expander(
                        "🔍 Agent Decision Details"
                    ):

                        st.write(
                            f"**Tool:** "
                            f"`{decision.tool}`"
                        )

                        st.write(
                            f"**Answerable:** "
                            f"`{decision.answerable}`"
                        )

                        if decision.reason:

                            st.write(
                                f"**Reason:** "
                                f"{decision.reason}"
                            )

                        if decision.operation:

                            st.write(
                                f"**Operation:** "
                                f"`{decision.operation}`"
                            )

                        if decision.value_column:

                            st.write(
                                f"**Value Column:** "
                                f"`{decision.value_column}`"
                            )

                        if decision.group_column:

                            st.write(
                                f"**Group Column:** "
                                f"`{decision.group_column}`"
                            )

                        if decision.filter_column:

                            st.write(
                                f"**Filter Column:** "
                                f"`{decision.filter_column}`"
                            )


            # ==================================
            # ERROR HANDLING
            # ==================================

            except Exception as error:

                st.error(
                    "Something went wrong while "
                    "processing your question."
                )

                with st.expander(
                    "Technical details"
                ):

                    st.exception(
                        error
                    )