# Skin Cancer SVM Classifier

A machine learning pipeline that classifies skin lesion images into 7 diagnostic categories using a Support Vector Machine (SVM), trained on the [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). Includes a Flask REST API for inference.

---

## Features

- Classifies 7 types of skin lesions with risk level indicators
- StandardScaler → PCA → SVM pipeline with class-imbalance handling
- Flask REST API for single-image inference
- Generates a confusion matrix and classification report after training

## Lesion Classes

| Code | Diagnosis | Risk |
|------|-----------|------|
| `mel` | Melanoma | HIGH |
| `akiec` | Actinic Keratoses and Intraepithelial Carcinoma | MODERATE |
| `bcc` | Basal Cell Carcinoma | MODERATE |
| `bkl` | Benign Keratosis-like Lesions | LOW |
| `df` | Dermatofibroma | LOW |
| `nv` | Melanocytic Nevi | LOW |
| `vasc` | Vascular Lesions | LOW |

---

## Project Structure

```
.
├── model.py              # Training pipeline
├── api.py                # Flask REST API
├── requirements.txt      # Python dependencies
├── data/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/   # *.jpg
│   └── HAM10000_images_part_2/   # *.jpg
└── outputs/              # Generated after training
    ├── svm_skin_cancer.pkl
    ├── label_encoder.pkl
    ├── confusion_matrix.png
    └── classification_report.txt
```

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Download the dataset**

Download the HAM10000 dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place the files under `data/` as shown in the structure above.

---

## Training

```bash
python model.py
```

This will:
1. Load and index all images from `data/`
2. Preprocess images (resize to 64×64, flatten, normalize)
3. Encode labels and compute class weights for imbalance handling
4. Train a StandardScaler → PCA (100 components) → SVM (RBF kernel) pipeline
5. Evaluate on a held-out 20% test set
6. Save the model and label encoder to `outputs/`

**Key configuration options** (edit at the top of `model.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `IMG_SIZE` | `(64, 64)` | Image resize dimensions |
| `N_COMPONENTS` | `100` | PCA components |
| `TEST_SIZE` | `0.2` | Fraction held out for testing |
| `USE_GRID_SEARCH` | `False` | Enable hyperparameter tuning (slow) |
| `MAX_SAMPLES` | `None` | Cap dataset size for quick runs |

**Quick test run** — set `MAX_SAMPLES = 2000` in `model.py` to train on a small subset.

---

## Running the API

Requires a trained model in `outputs/` (run training first).

```bash
python api.py
```

The server starts on `http://0.0.0.0:5000`.

### Endpoints

#### `GET /health`

Returns model load status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "outputs/svm_skin_cancer.pkl",
  "error": null
}
```

#### `POST /predict`

Upload a skin lesion image and receive a classification result.

**Request:** `multipart/form-data` with an `image` field.

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/lesion.jpg"
```

**Response:**

```json
{
  "modelLoaded": true,
  "predictedClass": "mel",
  "diagnosis": "Melanoma",
  "confidence": 0.823,
  "riskLevel": "HIGH",
  "probabilities": {
    "akiec": 0.021,
    "bcc": 0.034,
    "bkl": 0.018,
    "df": 0.009,
    "mel": 0.823,
    "nv": 0.082,
    "vasc": 0.013
  }
}
```

---

## Programmatic Inference

You can also call the prediction helper directly in Python:

```python
from model import predict_single_image

result = predict_single_image("path/to/image.jpg")
print(result["label"])        # e.g. "Melanoma"
print(result["probabilities"]) # dict of all class probabilities
```

---

## Model Details

| Component | Detail |
|-----------|--------|
| Preprocessing | Resize to 64×64, flatten to 12,288 features, normalize to [0, 1] |
| Scaler | `StandardScaler` |
| Dimensionality reduction | PCA (100 components) |
| Classifier | SVM, RBF kernel, C=10, `class_weight='balanced'` |
| Imbalance handling | Per-class weights via `compute_class_weight` |
| Probability output | Enabled via Platt scaling (`probability=True`) |

---

## ⚠️ Disclaimer

This tool is intended for **research and educational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified dermatologist for clinical evaluation of skin lesions.