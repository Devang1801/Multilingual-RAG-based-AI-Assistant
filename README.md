# Internship - Multilingual RAG-based AI Assistant

An intelligent, multilingual conversational agent designed to help users query internship program details. It uses a Retrieval-Augmented Generation (RAG) architecture powered by a local Large Language Model (Qwen-4B) to answer general questions and retrieve user-specific application and internship details.

## 🏗️ System Architecture

![System Architecture](System%20Architecture%20-%20Multilingual%20RAG%20AI%20Assistant.png)

## 🚀 Features

- **Multilingual Chat Interface**: Supports querying in multiple languages and Hinglish (e.g., "stipend kab aayega?", "what is my application status?").
- **Local RAG Pipeline**: Uses `FAISS` and `HuggingFaceEmbeddings` (`all-mini_v12`) to perform semantic search over program guidelines stored in Markdown.
- **Context-Aware Intent Classification**: Intelligently routes user queries between "General Knowledge" (program rules, eligibility) and "User-Specific Data" (stipend, DBT status, grievances).
- **Mock Authentication & User API**: Simulates real-world DB checks and OTP sign-ins via a local FastAPI service.
- **Data Privacy & Local LLM**: Runs entirely locally using `Qwen-4B` and `PyTorch`, ensuring sensitive candidate data is not sent to external APIs.

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python, Uvicorn
- **AI/ML**: PyTorch, Transformers, HuggingFace (`all-mini_v12`, `Qwen-4B`)
- **RAG & NLP**: LangChain, FAISS, Rank-BM25
- **Frontend**: Jinja2 Templates, HTML/CSS/JS

## 📁 Project Structure

```text
Internship-AI_Assistant/
├── agents.py                # Main FastAPI Server & Chatbot Logic
├── mock_user_api.py         # Mock API for OTP verification and user details
├── ingest.py                # Script to convert Markdown to FAISS Vector DB
├── markdown/                # Contains the knowledge base (e.g., Merged.md)
├── templates/               # HTML Templates for the chat UI
├── static/                  # CSS/JS assets for the frontend
├── requirements_6nov.txt    # Python dependencies
└── vectorstore/             # FAISS index and doc_summary.json (Generated)
```

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed and a working virtual environment.

### 2. Install Dependencies
```bash
pip install -r requirements_6nov.txt
```

### 3. Build the Vector Database
Ingest the markdown knowledge base into FAISS:
```bash
python ingest.py
```
*(This will read `.md` files from the `markdown/` folder and generate the `vectorstore/` directory).*

### 4. Run the Mock API Server
In a new terminal, start the mock authentication and user details server:
```bash
python mock_user_api.py
```
*(Runs on `http://localhost:8900`. Use test mobile `7668455121` and OTP `123456` to login).*

### 5. Run the Main Agent Server
In another terminal, start the main chatbot application:
```bash
python agents.py
```
*(Runs on `http://localhost:8000` by default).*

### 6. Access the Application
Open your browser and navigate to:
[http://localhost:8000/](http://localhost:8000/)

## 📝 License
This project is licensed under the MIT License.
