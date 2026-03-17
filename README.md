# LOTR RAG

A Retrieval-Augmented Generation app that lets you ask questions about the Lord of the Rings books.

## Tech Stack

- Python 3.12
- Flask (web server)
- LangChain (RAG pipeline)
- FAISS (vector search)
- HuggingFace sentence-transformers (embeddings)
- OpenAI GPT-4o-mini (LLM)
- Gunicorn (production server)
- Railway (deployment)

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

## Usage

### 1. Ingest PDFs

```bash
python -m src.ingest
```

Reads the LOTR PDFs from `data/pdf/`, chunks them, and saves a FAISS vector store to `vectorstore/`.

### 2. Run the web app

```bash
python -m src.app
```

Open `http://localhost:5000` in your browser.

## Project Structure

```
src/
  config.py    — settings and env loading
  rag.py       — embeddings, vectorstore, chain logic
  ingest.py    — PDF ingestion pipeline
  app.py       — Flask web server
templates/     — HTML frontend
static/images/ — background image
data/pdf/      — source PDFs
vectorstore/   — FAISS index (generated)
```

## Deployment

Deployed on Railway. Set `OPENAI_API_KEY` in Railway's environment variables.
