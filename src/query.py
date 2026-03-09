from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "vectorstore")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k":4})

llm = ChatOpenAI(model="gpt-4o-mini")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

while True:
    q = input("Question: ")
    result = qa.invoke(q)
    print("\nAnswer:\n", result["result"], "\n")