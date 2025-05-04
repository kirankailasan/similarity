from flask import Flask, request, jsonify
import numpy as np
from sentence_transformers import SentenceTransformer, util
import json

app = Flask(__name__)

# Load precomputed QA pairs and embeddings
QA_PAIRS = None
QA_EMBEDDINGS = None
hf_model = None

def preload():
    global QA_PAIRS, QA_EMBEDDINGS, hf_model
    if QA_PAIRS is None:
        with open("qa.json", encoding="utf-8") as f:
            QA_PAIRS = json.load(f)
    if QA_EMBEDDINGS is None:
        QA_EMBEDDINGS = np.load("qa.npy")
    if hf_model is None:
        hf_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("QA system preloaded and ready.")

@app.route('/find-similar', methods=['POST'])
def find_similar_answer():
    data = request.json
    user_question = data.get('question')

    if not user_question:
        return jsonify({'error': 'No question provided'}), 400

    # Compute embedding for the user's question
    user_embedding = hf_model.encode(user_question, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, QA_EMBEDDINGS)[0]
    best_score = float(similarities.max())
    best_index = int(similarities.argmax())

    if best_score >= 0.6:  # threshold for similarity
        return jsonify({'answer': QA_PAIRS[best_index]["answer"]})
    else:
        return jsonify({'answer': 'Sorry, I could not find a relevant answer.'})

if __name__ == '__main__':
    preload()  # Preload the model and data
    app.run(host='0.0.0.0', port=5000)
