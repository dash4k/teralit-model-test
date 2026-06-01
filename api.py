"""
Teralit Skin Disease Classifier — Flask API (YOLO backend)
Datasets: https://github.com/Liyang-A-O/Streamlit-Teralit
Endpoint: POST /predict
Input  : multipart/form-data with field "image" (jpg/png/webp)
Output : JSON with modelLoaded, predictedClass, diagnosis,
         confidence, riskLevel, probabilities
"""

import os
import io
import logging
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from ultralytics import YOLO

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = os.getenv("MODEL_DIR", "saved_models")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "10"))
IMG_SIZE    = int(os.getenv("IMG_SIZE", "224"))
RISK_MAP = {
    "high":   ["melanoma"],
    "medium": ["chickenpox", "eczema", "hives"],
    "low":    [],
}


def get_risk_level(class_name: str) -> str:
    name_lower = class_name.lower()
    for level in ("high", "medium", "low"):
        for keyword in RISK_MAP[level]:
            if keyword in name_lower:
                return level
    return "medium"


# ── Model loading ──────────────────────────────────────────────────────────────
_model       = None
_class_names = []
_num_classes = 0
_model_ok    = False


def load_model_assets():
    global _model, _class_names, _num_classes, _model_ok

    import glob
    pt_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pt")))
    if not pt_files:
        logger.error("No .pt model found in %s", MODEL_DIR)
        return
    pt_path = pt_files[-1]  # use the latest if multiple exist

    logger.info("Loading YOLO model from %s …", pt_path)
    _model = YOLO(pt_path)
    logger.info("YOLO model loaded.")

    raw_names = _model.names
    if isinstance(raw_names, dict):
        _num_classes = len(raw_names)
        _class_names = [raw_names[i] for i in range(_num_classes)]
    else:
        _class_names = list(raw_names)
        _num_classes = len(_class_names)

    _model_ok = True
    logger.info("Ready. img_size=%d classes=%s", IMG_SIZE, _class_names)


load_model_assets()


# ── /predict ───────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    model_loaded = _model_ok

    if "image" not in request.files:
        return jsonify({"modelLoaded": model_loaded, "predictedClass": None,
                        "diagnosis": None, "confidence": None,
                        "riskLevel": None, "probabilities": None,
                        "error": "No 'image' field in request."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"modelLoaded": model_loaded, "predictedClass": None,
                        "diagnosis": None, "confidence": None,
                        "riskLevel": None, "probabilities": None,
                        "error": "Empty filename."}), 400

    file.seek(0, 2); size_mb = file.tell() / 1e6; file.seek(0)
    if size_mb > MAX_FILE_MB:
        return jsonify({"modelLoaded": model_loaded, "predictedClass": None,
                        "diagnosis": None, "confidence": None,
                        "riskLevel": None, "probabilities": None,
                        "error": f"File too large ({size_mb:.1f} MB). Max {MAX_FILE_MB} MB."}), 413

    if not _model_ok:
        return jsonify({"modelLoaded": False, "predictedClass": None,
                        "diagnosis": None, "confidence": None,
                        "riskLevel": None, "probabilities": None,
                        "error": "Model not loaded. Check server logs."}), 503

    try:
        pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"modelLoaded": model_loaded, "predictedClass": None,
                        "diagnosis": None, "confidence": None,
                        "riskLevel": None, "probabilities": None,
                        "error": f"Cannot open image: {e}"}), 422

    try:
        results = _model.predict(source=pil_img, imgsz=IMG_SIZE,
                                 verbose=False)
        result = results[0]

        probs_tensor = result.probs.data.cpu().numpy()

        pred_idx   = int(np.argmax(probs_tensor))
        pred_class = _class_names[pred_idx]
        confidence = float(probs_tensor[pred_idx])

        probabilities = {
            _class_names[i]: round(float(probs_tensor[i]), 6)
            for i in range(_num_classes)
        }

        return jsonify({
            "modelLoaded":    True,
            "predictedClass": pred_class,
            "diagnosis":      pred_class,
            "confidence":     confidence,
            "riskLevel":      get_risk_level(pred_class),
            "probabilities":  probabilities,
        })

    except Exception as e:
        logger.exception("Inference error")
        return jsonify({"modelLoaded": model_loaded, "predictedClass": None,
                        "diagnosis": None, "confidence": None,
                        "riskLevel": None, "probabilities": None,
                        "error": f"Inference failed: {e}"}), 500


# ── /health ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":      "ok" if _model_ok else "degraded",
        "modelLoaded": _model_ok,
        "numClasses":  _num_classes,
        "classNames":  _class_names,
        "imgSize":     [IMG_SIZE, IMG_SIZE],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)