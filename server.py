# Author: Mehmet Can Odabas

# Explanation:
# Exposes the trained MNIST model as a small local web API for inference.
# The lab assignment does not require an API; it only performs training and testing in the terminal.
# Endpoints: POST /predict, GET /health
# Preprocessing: grayscale → 28x28 → normalize [-1,1] → flatten (1,784).
# CORS enabled so a local HTML page (draw.html) can call this API.

# Run: python server.py
# Then open in your browser (health check):
# http://127.0.0.1:5000/health
# Then open the draw.html page to use the drawing interface.

import base64
import io
import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from flask_cors import CORS  # allow calls from file:// or other origins

# Path to the trained model (produced by the lab's training script)
MODEL_PATH = r"C:\Users\Mehmet Can Odabas\Desktop\GitHub\computer-vision-mnist-nn\mnist_mlp.keras"

# Load once at startup
model = tf.keras.models.load_model(MODEL_PATH)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # simple permissive CORS

def normalize(x: np.ndarray) -> np.ndarray:
    """0..255 → [-1, 1]."""
    return (x.astype("float32") / 255.0) * 2.0 - 1.0

def preprocess(img: Image.Image) -> np.ndarray:
    """PNG from canvas → model input."""
    img = img.convert("L").resize((28, 28), Image.LANCZOS)
    arr = normalize(np.array(img))
    return arr.reshape(1, -1)  # (1, 784)

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(silent=True)
    if not data or "image_base64" not in data:
        return jsonify({"error": "image_base64 is required"}), 400

    try:
        img_bytes = base64.b64decode(data["image_base64"])
        img = Image.open(io.BytesIO(img_bytes))
        x = preprocess(img)
        probs = model.predict(x, verbose=0)[0]  # (10,)
        pred = int(np.argmax(probs))
        return jsonify({
            "prediction": pred,
            "probabilities": [float(p) for p in probs]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # For local use; change host to "0.0.0.0" to expose on LAN.
    app.run(host="127.0.0.1", port=5000, debug=True)
