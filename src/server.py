from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
import torch

from attention_core import load_vocab, load_model, predict_with_self_attention

app = Flask(__name__, static_folder=str(Path(__file__).resolve().parent / "frontend"), static_url_path="/static")

# Global model/vocab/device
vocab = load_vocab()
model, device = load_model(vocab)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text kosong"}), 400
    prob_pos, tokens, attn = predict_with_self_attention(model, vocab, text, device=device)
    return jsonify(
        {
            "prob_positive": prob_pos,
            "tokens": tokens,
            "attention": attn.tolist(),
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
