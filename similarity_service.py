import numpy as np
import json
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

with open("qa.json", encoding="utf-8") as f:
    QA_PAIRS = json.load(f)

QA_EMBEDDINGS = np.load("qa.npy")
hf_model = SentenceTransformer('all-MiniLM-L6-v2')

def find_similar(question):
    user_embedding = hf_model.encode(question, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, QA_EMBEDDINGS)[0]
    best_score = float(similarities.max())
    best_index = int(similarities.argmax())
    if best_score >= 0.6:
        return QA_PAIRS[best_index]["answer"]
    else:
        return None 

@app.route("/", methods=["POST"])
def similarity_api():
    data = request.get_json()
    questions = data.get("data", [])
    answers = [find_similar(q) for q in questions]
    return jsonify({"data": answers})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
