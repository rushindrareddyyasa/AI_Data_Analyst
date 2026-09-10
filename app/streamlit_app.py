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


# ==================================================
# STREAMLIT PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
)


# ==================================================
# APPLICATION HEADER
# ==================================================

st.title("🤖 AI Data Analyst")

st.write(
    "Upload a dataset and interact with it using "
    "natural language."
)


# ==================================================
# DATASET UPLOAD
# ==================================================

st.sidebar.header("📁 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload a dataset",
    type=["csv", "xlsx"],
    help="Upload a CSV or Excel dataset.",
)


# ==================================================
# LOAD DATASET
# ==================================================

if uploaded_file is not None:

    try:

        df = load_dataset(uploaded_file)

        dataset_name = uploaded_file.name

        st.sidebar.success(
            f"Loaded: {dataset_name}"
        )

    except Exception as error:

        st.sidebar.error(
            f"Unable to load dataset: {error}"
        )

        st.stop()

else:

    default_dataset = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "sales.csv"
    )

    try:

        df = load_dataset(default_dataset)

        dataset_name = "sales.csv"

        st.sidebar.info(
            "Using default dataset"
        )

    except Exception as error:

        st.error(
            f"Unable to load default dataset: {error}"
        )

        st.stop()


# ==================================================
# DATASET VALIDATION
# ==================================================

if df.empty:

    st.error(
        "The uploaded dataset is empty."
    )

    st.stop()


# ==================================================
# CREATE DATABASE
# ==================================================

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "data.db"
)

engine = create_database(
    df=df,
    database_path=str(DATABASE_PATH),
    table_name="dataset",
)


# ==================================================
# SIDEBAR DATASET INFORMATION
# ==================================================

with st.sidebar:

    st.divider()

    st.header("📊 Dataset Information")

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
# MAIN DATASET PREVIEW
# ==================================================

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True,
)


# ==================================================
# QUESTION SECTION
# ==================================================

st.subheader("💬 Ask Your Dataset")

question = st.text_input(
    "Ask a question",
    placeholder=(
        "Example: What is the average value by category?"
    ),
)


# ==================================================
# ASK AI
# ==================================================

if st.button(
    "🚀 Ask AI",
    type="primary",
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

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

                st.subheader("💡 Answer")

                st.write(
                    result["answer"]
                )


                # ==================================
                # TOOL USED
                # ==================================

                st.subheader("🔧 Analysis Method")

                st.info(
                    result["tool"].upper()
                )


                # ==================================
                # ANALYSIS RESULT
                # ==================================

                if result["result"] is not None:

                    st.subheader(
                        "📋 Analysis Result"
                    )

                    analysis_result = result["result"]

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

                    for source in result["sources"]:

                        st.write(
                            f"• {source}"
                        )


            except Exception as error:

                st.error(
                    "Something went wrong while "
                    "processing your question."
                )

                with st.expander(
                    "Technical details"
                ):

                    st.exception(error)