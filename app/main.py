import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from ai_data_analyst.ingestion.dataset import load_dataset
from ai_data_analyst.sql.database import create_database
from ai_data_analyst.agent import run_agent


app = FastAPI(
    title="AI Data Analyst API",
    description="AI-powered data analysis using SQL, Python, Visualization, and RAG.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "AI Data Analyst API is running",
        "status": "online",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    try:
        # Read uploaded file
        file_bytes = await file.read()

        # Convert bytes into a file-like object
        from io import BytesIO

        uploaded_file = BytesIO(file_bytes)
        uploaded_file.name = file.filename

        # Load CSV/XLSX
        df = load_dataset(uploaded_file)

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded dataset is empty.",
            )

        # Create SQLite database
        database_path = (
            PROJECT_ROOT / "data" / "processed" / "data.db"
        )

        engine = create_database(
            df=df,
            database_path=":memory:",
            table_name="dataset",
        )

        # Run existing AI agent
        result = run_agent(
            df=df,
            engine=engine,
            question=question,
        )

        # Convert result into JSON-safe response
        response = {
            "question": question,
            "tool": result["tool"],
            "answer": result["answer"],
            "answerable": result["decision"].answerable,
        }

        # Include analysis result when available
        if result["result"] is not None:
            analysis_result = result["result"]

            if isinstance(analysis_result, pd.DataFrame):
                response["result"] = analysis_result.to_dict(
                    orient="records"
                )
            else:
                response["result"] = analysis_result

        # Include visualization information
        if result["figure"] is not None:
            response["visualization"] = True
        else:
            response["visualization"] = False

        # Include RAG sources
        if result["sources"]:
            response["sources"] = result["sources"]
        else:
            response["sources"] = []

        return response

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(error)}",
        )