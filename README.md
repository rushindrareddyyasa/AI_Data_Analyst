# 🩺 MediBot – AI-Powered Medical RAG Chatbot

MediBot is a Retrieval-Augmented Generation (RAG) based medical chatbot that answers health-related questions by retrieving relevant information from medical documents and generating responses using Google Gemini.

This project combines Large Language Models (LLMs), vector databases, and document retrieval to build a reliable domain-specific assistant.

---

## 🚀 Features

- 📚 PDF-based Medical Knowledge Base
- 🤖 Powered by Google Gemini
- 🧠 Semantic Search using Pinecone
- 🔍 Retrieval-Augmented Generation (RAG)
- 🌐 Web Interface using Flask
- ⚡ Fast & Scalable Architecture
- 🔐 Secure API Key Management

---

## 🏗️ Tech Stack

| Technology | Usage |
|------------|--------|
| Python | Core Backend |
| Flask | Web Server |
| LangChain | RAG Pipeline |
| Google Gemini | LLM |
| Pinecone | Vector Database |
| HuggingFace | Embeddings |
| HTML / CSS / JS | Frontend |
| Git & GitHub | Version Control |

---

## 📁 Project Structure

```text
Rag_medical_chatbot/
│
├── app.py                 # Main Flask application
├── store_index.py         # PDF ingestion & Pinecone indexing
├── setup.py               # Project packaging configuration
├── requirements.txt       # Project dependencies
├── README.md              # Documentation
│
├── src/
│   ├── __init__.py
│   ├── helper.py          # PDF loading & embeddings
│   └── prompt.py
│
├── data/
│   └── Medical_book.pdf   # Medical reference PDF
│
├── templates/
│   └── index.html         # HTML template
│
├── static/
│   ├── css/
│   └── js/
│
├── .env.example           # Environment template
└── .gitignore
```



---

## 🧠 System Architecture

User → Flask → Pinecone → Relevant Docs → Gemini → Response


### Workflow:

1. Load medical PDFs
2. Split into text chunks
3. Generate embeddings
4. Store in Pinecone
5. Retrieve relevant chunks
6. Generate answer using Gemini
7. Display result

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd Rag_medical_chatbot
```

### 2️⃣ Create Virtual Environment

Using Conda:

```bash
conda create -n medibot python=3.10
conda activate medibot
```
OR using venv:

```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_REGION=us-east-1
```

### 5️⃣ Build Vector Database

Before running the chatbot, index documents:

```bash
python store_index.py
```

### 6️⃣ Run Application

```bash
python app.py
```

---

## 💬 Usage

Ask medical questions through the web interface.
Examples:
- What is diabetes?
- What are dengue symptoms?
- How to prevent malaria?
- What is hypertension?

---

## 🛡️ Disclaimer

This chatbot is not a substitute for professional medical advice.

Always consult a certified healthcare professional.

---

## 👨‍💻 Author

Rusheenddra Basani










