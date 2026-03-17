from src import config  # noqa: F401
from src.rag import build_chain

chain = build_chain()

if __name__ == "__main__":
    while True:
        q = input("Question: ")
        result = chain.invoke(q)
        print("\nAnswer:\n", result, "\n")
