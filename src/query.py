from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

DB_PATH = "vectorstore"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
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
    answer = qa.run(q)
    print(answer)