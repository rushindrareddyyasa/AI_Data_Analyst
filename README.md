# 🤖 AI Data Analyst

AI Data Analyst is an LLM-powered data analysis application that allows users to upload datasets and ask questions about them using natural language.

Instead of requiring users to write SQL or Python manually, the application understands the question, checks whether the uploaded dataset can answer it, selects the appropriate analysis method, executes the analysis, and returns the result.

The project also includes a RAG pipeline for answering questions from unstructured documents and a FastAPI backend for exposing the system as an API.

---

## 🚀 Features

- 📂 Upload CSV and Excel datasets
- 🧠 LLM-based question understanding and routing
- 🛡️ Dataset-aware answerability checking
- 🗃️ Natural-language to SQL analysis
- 🐍 Python/Pandas-based data analysis
- 📊 Natural-language-driven visualizations
- 📚 RAG for unstructured documents
- 🔎 Semantic document retrieval
- 🧩 ChromaDB vector database
- ⚡ FastAPI backend
- 🌐 Streamlit interface
- 🔐 Environment-based API key management
- 🧪 Automated tests using Pytest

---

## 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | Core application |
| Google Gemini | LLM and embeddings |
| FastAPI | Backend API |
| Streamlit | User interface |
| Pandas / NumPy | Data analysis |
| SQLAlchemy | Database interaction |
| SQLite | Temporary structured-data querying |
| ChromaDB | Vector database |
| Matplotlib | Data visualization |
| Pytest | Testing |
| uv | Python dependency management |
| Git / GitHub | Version control |

---

## 📁 Project Structure

```text
AI_Data_Analyst/
│
├── app/
│   ├── main.py                 # FastAPI application
│   └── streamlit_app.py        # Streamlit application
│
├── data/
│   └── raw/
│       ├── sales.csv           # Sample structured dataset
│       └── documents/
│           └── regional_sales_policy.txt
│
├── src/
│   └── ai_data_analyst/
│       ├── agent.py            # Main AI agent
│       │
│       ├── analysis/
│       │   ├── analyzer.py
│       │   ├── executor.py
│       │   └── filter.py
│       │
│       ├── ingestion/
│       │   ├── dataset.py      # CSV/XLSX loading
│       │   ├── loader.py
│       │   └── profiler.py
│       │
│       ├── llm/
│       │   ├── client.py
│       │   ├── prompts.py
│       │   ├── response.py
│       │   ├── router.py       # Question routing
│       │   └── validator.py
│       │
│       ├── rag/
│       │   ├── chunker.py
│       │   ├── embeddings.py
│       │   ├── generator.py
│       │   ├── ingest.py
│       │   ├── loader.py
│       │   ├── pipeline.py
│       │   ├── retriever.py
│       │   └── vector_store.py
│       │
│       ├── sql/
│       │   ├── database.py
│       │   ├── executor.py
│       │   ├── generator.py
│       │   ├── query_engine.py
│       │   ├── schema.py
│       │   └── validator.py
│       │
│       └── visualization/
│           ├── charts.py
│           └── executor.py
│
├── tests/                      # Unit and integration tests
│
├── .env                        # Local secrets (not committed)
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 🧠 System Architecture

```text
                         User
                           │
                           ▼
                    Streamlit / API
                           │
                           ▼
                    AI Agent / Router
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
            SQL          Python     Visualization
             │             │             │
             ▼             ▼             ▼
          SQLite         Pandas       Matplotlib
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                        Response


                 Unstructured Documents
                           │
                           ▼
                       RAG Pipeline
                           │
                    ┌──────┴──────┐
                    ▼             ▼
                Embeddings     Chunks
                    │             │
                    └──────┬──────┘
                           ▼
                        ChromaDB
                           │
                           ▼
                       Retriever
                           │
                           ▼
                         Gemini
                           │
                           ▼
                    Answer + Sources
```

---

## 🔄 How the Project Works

### 1. Dataset Upload

The user uploads a CSV or Excel file through the application.

The ingestion layer:

- Loads the dataset
- Detects supported file formats
- Handles common CSV encodings
- Detects date/time columns
- Creates a Pandas DataFrame

---

### 2. Dynamic Dataset Understanding

The system does not assume that every dataset contains sales-related columns.

It dynamically identifies:

- Dataset columns
- Data types
- Numerical columns
- Categorical columns
- Date columns

This information is provided to the LLM router.

---

### 3. Question Routing

The user asks a natural-language question.

The LLM router determines which capability should handle it:

```text
                    User Question
                          │
                          ▼
                    LLM Router
                          │
        ┌─────────┬───────┼────────┬─────────┐
        ▼         ▼       ▼        ▼         ▼
       SQL      Python  Chart      RAG      None
```

`None` is used when the dataset or available documents cannot answer the question.

---

### 4. Answerability Check

Before executing an analysis, the system checks whether the requested information is represented in the dataset.

For example, if a tourism dataset contains 92 destinations and the user asks:

```text
What is the employee count?
```

the system does **not** assume:

```text
92 rows = 92 employees
```

Instead, it returns:

```text
The dataset does not contain employee information.
```

This prevents incorrect answers caused by confusing row counts with real-world entity counts.

---

### 5. SQL Analysis

For structured questions that are suitable for SQL:

```text
Natural Language Question
          ↓
Dynamic Schema
          ↓
LLM SQL Generator
          ↓
SQL Validator
          ↓
SQLite
          ↓
Query Result
          ↓
LLM Explanation
```

Example:

```text
What are total sales by region?
```

The SQL layer generates and validates a query using the actual uploaded dataset schema.

---

### 6. Python Analysis

Questions that are better suited to dataframe operations are handled using Python and Pandas.

Example:

```text
What is the average sales value?
```

The analysis executor performs the calculation and the response layer converts the result into a natural-language answer.

---

### 7. Visualization

The application can generate visualizations when the question requires a chart.

Example:

```text
Show sales by region.
```

The visualization layer determines the required chart workflow and generates the chart using Matplotlib.

---

### 8. RAG Pipeline

Questions related to unstructured documents are handled through Retrieval-Augmented Generation.

```text
Document
   ↓
Document Loader
   ↓
Text Chunking
   ↓
Embedding Generation
   ↓
ChromaDB
   ↓
Similarity Retrieval
   ↓
Relevant Context
   ↓
Gemini
   ↓
Answer + Sources
```

This allows the application to combine structured dataset analysis with document-based question answering.

---

## 🔬 Development Process

The project was developed in multiple stages.

### Phase 1 — Data Ingestion

Started with a structured sales dataset and implemented:

- CSV loading
- Date detection
- Dataset profiling
- Basic validation

### Phase 2 — Generic Dataset Support

The system was extended to support arbitrary uploaded datasets instead of relying on fixed sales columns.

CSV and XLSX uploads were added.

### Phase 3 — Dynamic Schema

A schema generation layer was introduced so that SQL generation and routing could use the actual structure of the uploaded dataset.

### Phase 4 — SQL Engine

Implemented:

- SQLite database creation
- Dynamic SQL generation
- SQL validation
- SQL execution
- Natural-language result explanation

### Phase 5 — Python Analysis

Added Pandas-based analytical operations for calculations and grouped analysis.

### Phase 6 — Visualization

Added a visualization executor and chart generation workflows.

### Phase 7 — RAG

Built the document pipeline:

```text
Load → Chunk → Embed → Store → Retrieve → Generate
```

ChromaDB was introduced for vector storage.

### Phase 8 — Agent

Integrated SQL, Python, Visualization, and RAG behind a single AI agent.

### Phase 9 — Reliability Improvements

Testing revealed that an LLM could incorrectly interpret the number of dataset rows as an answer to an unrelated entity-count question.

The router was enhanced with an explicit answerability decision and dataset-aware reasoning.

### Phase 10 — FastAPI

The agent was exposed through FastAPI so the same AI analysis system can be consumed through HTTP APIs.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI_Data_Analyst
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the GitHub repository URL.

---

### 2. Install uv

Install `uv` if it is not already available on your system.

Verify:

```bash
uv --version
```

---

### 3. Install Project Dependencies

From the project root:

```bash
uv sync
```

This uses `pyproject.toml` and `uv.lock` to install the project dependencies.

---

### 4. Configure the Gemini API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

The `.env` file is ignored by Git and should never be committed.

---

## ▶️ Run the Streamlit Application

Start the Streamlit interface:

```bash
uv run streamlit run app/streamlit_app.py
```

Then open the URL displayed by Streamlit, normally:

```text
http://localhost:8501
```

### Using the Application

1. Upload a `.csv` or `.xlsx` dataset.
2. Wait for the dataset preview to load.
3. Enter a natural-language question.
4. Click **Ask AI**.
5. The agent selects the appropriate analysis method.
6. View the answer, analysis result, visualization, or RAG sources.

Example questions:

```text
What are total sales by region?
```

```text
What is the average sales?
```

```text
Show sales by region.
```

```text
What information is available in this dataset?
```

---

## ⚡ Run the FastAPI Backend

Start the FastAPI server:

```bash
uv run fastapi dev app/main.py
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

---

## 📖 FastAPI Documentation

FastAPI provides interactive Swagger documentation at:

```text
http://127.0.0.1:8000/docs
```

From Swagger UI, the available endpoints can be tested directly.

---

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/analyze` | Analyze an uploaded dataset |

### `/`

Example response:

```json
{
  "message": "AI Data Analyst API is running",
  "status": "online"
}
```

### `/health`

Example response:

```json
{
  "status": "healthy"
}
```

### `/analyze`

Accepts:

```text
file     → CSV/XLSX dataset
question → Natural-language question
```

Example:

```text
Question:
What is the employee count?
```

If the uploaded dataset does not contain employee information, the API returns a response similar to:

```json
{
  "question": "What is the employee count?",
  "tool": "none",
  "answer": "The dataset does not contain employee information.",
  "answerable": false,
  "visualization": false,
  "sources": []
}
```

---

## 🧪 Testing

The project uses Pytest for automated testing.

Run:

```bash
uv run pytest -q
```

The tests cover components such as:

- Dataset ingestion
- Dataset profiling
- Analysis
- Filtering
- SQL generation
- SQL validation
- SQL execution
- LLM routing
- Route validation
- RAG
- Retrieval
- Vector storage
- Agent behavior
- Visualization

---

## 🔐 Security & Reliability

The project includes several mechanisms to improve reliability:

### Environment Variables

API credentials are stored in `.env` rather than source code.

### Dynamic Schema

The LLM receives the actual uploaded dataset schema.

### Answerability Validation

Questions unrelated to the available data are rejected instead of forcing an answer.

### SQL Validation

Generated SQL is validated before execution.

### Controlled Tool Routing

The LLM selects from predefined analysis capabilities rather than directly executing arbitrary application logic.

### RAG Sources

Document-based responses can expose the retrieved source documents used for the answer.

---

## 🔮 Future Improvements

- Automatic dataset profiling and insight generation
- Multi-turn conversations
- Conversation memory
- More chart types
- Improved retrieval relevance filtering
- PostgreSQL support
- Authentication and authorization
- API rate limiting
- Logging and observability
- LLM evaluation and monitoring
- Docker deployment
- Cloud deployment
- CI/CD pipeline
- Production-grade vector database
- Multi-user session management

---

## 👨‍💻 Author

**Rushindra Reddy**

B.Tech — Computer Science & Engineering

---

## ⭐ Project Objective

The main objective of this project is to demonstrate how an AI-powered analytical system can combine:

```text
LLMs
+
SQL
+
Python
+
Data Visualization
+
RAG
+
Vector Databases
+
AI Agents
+
FastAPI
+
Testing
```

into a single practical application that allows users to interact with data using natural language.
