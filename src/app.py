from src import config  # noqa: F401
from flask import Flask, render_template, request, jsonify
from src.rag import build_chain
from src.config import TEMPLATE_DIR, STATIC_DIR

chain = build_chain()

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        answer = chain.invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
