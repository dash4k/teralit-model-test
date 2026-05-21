"""
flask_api.py – Skin Cancer SVM Classifier (Flask)
==================================================
Single-file REST API for testing the trained SVM model.

Endpoints:
    GET  /health       – model status
    POST /predict      – upload one image, get prediction

Run:
    pip install flask pillow scikit-learn numpy joblib
    python flask_api.py
"""

import io
import joblib
import numpy as np
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "outputs/svm_skin_cancer.pkl"
ENCODER_PATH = "outputs/label_encoder.pkl"
IMG_SIZE     = (64, 64)

CLASS_LABELS = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}

RISK_LEVEL = {
    "mel":   "HIGH",
    "akiec": "MODERATE",
    "bcc":   "MODERATE",
    "bkl":   "LOW",
    "df":    "LOW",
    "nv":    "LOW",
    "vasc":  "LOW",
}

# ── Load model at startup ──────────────────────────────────────────────────────
pipeline      = None
label_encoder = None
model_error   = None

try:
    pipeline      = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
except Exception as e:
    model_error = str(e)
    print(f"✗ Could not load model: {e}")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({
        "status":       "ok" if pipeline is not None else "degraded",
        "model_loaded": pipeline is not None,
        "model_path":   MODEL_PATH,
        "error":        model_error,
    })


@app.post("/predict")
def predict():
    # ── 1. Model ready? ────────────────────────────────────────────────────────
    if pipeline is None:
        return jsonify({
            "model_loaded": False,
            "error": f"Model not loaded: {model_error}"
        }), 503

    # ── 2. Image in request? ───────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send a file under the key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # ── 3. Preprocess ──────────────────────────────────────────────────────────
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB").resize(IMG_SIZE)
        x   = np.array(img, dtype=np.float32).flatten() / 255.0
    except Exception as e:
        return jsonify({"error": f"Could not process image: {e}"}), 422

    # ── 4. Inference ───────────────────────────────────────────────────────────
    try:
        predicted_class = pipeline.predict([x])[0]
        proba           = pipeline.predict_proba([x])[0]
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    # ── 5. Build response ──────────────────────────────────────────────────────
    prob_dict = {
        cls: round(float(p), 6)
        for cls, p in zip(label_encoder.classes_, proba)
    }

    return jsonify({
        "modelLoaded":    True,
        "predictedClass": predicted_class,
        "diagnosis":           CLASS_LABELS.get(predicted_class, predicted_class),
        "confidence":      round(float(proba.max()), 6),
        "riskLevel":      RISK_LEVEL.get(predicted_class, "UNKNOWN"),
        "probabilities":   prob_dict,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)