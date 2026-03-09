from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF_DIR = os.path.join(BASE_DIR, "data", "pdf")
DB_PATH = os.path.join(BASE_DIR, "vectorstore")

docs = []

for file in os.listdir(PDF_DIR):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(PDF_DIR, file))
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local(DB_PATH)

print("Ingestion complete.")