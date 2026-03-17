# LOTR RAG

A Retrieval-Augmented Generation app that lets you ask questions about the LOTR books. 

Runs fully locally —> no data leaves your machine.

## Tech Stack

- Python 3.12
- Flask (web server)
- LangChain (RAG pipeline)
- FAISS (vector search)
- HuggingFace sentence-transformers (embeddings)
- Ollama + Llama 3 (local LLM)

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- Pull the model: `ollama pull llama3`

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

## Usage

### 1. Ingest PDFs

```bash
python -m src.ingest
```

### 2. Run the web app

```bash
python -m src.app
```

Open `http://localhost:5000`

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
