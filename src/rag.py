from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import DB_PATH, EMBEDDING_MODEL, LLM_MODEL, OLLAMA_BASE_URL, RETRIEVER_K

PROMPT_TEMPLATE = (
    "Answer the question based on the following context:\n\n"
    "{context}\n\nQuestion: {question}"
)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_vectorstore(embeddings=None):
    embeddings = embeddings or get_embeddings()
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)


def build_chain(retriever=None):
    if retriever is None:
        db = load_vectorstore()
        retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_K})

    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
