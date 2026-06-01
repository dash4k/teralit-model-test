# Teralit Skin Disease Classifier

A YOLO-based computer vision model for classifying skin diseases using Flask REST API. This is the inference and testing service for the [Streamlit-Teralit](https://github.com/Liyang-A-O/Streamlit-Teralit) project.

---

## Features

- YOLO object detection and classification model
- Flask REST API for real-time skin disease prediction
- Risk level indicators (high, medium, low)
- Configurable via environment variables
- Support for jpg, png, and webp image formats

---

## Project Structure

```
.
├── api.py                # Flask REST API
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── saved_models/        # YOLO .pt model files (excluded from git)
    └── *.pt             # Latest model will be auto-loaded
```

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Prepare the model**

Place your trained YOLO `.pt` model file(s) in the `saved_models/` directory. The API will automatically load the latest model found.

---

## Running the API

```bash
python api.py
```

The server starts on `http://0.0.0.0:5000`.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `saved_models` | Directory containing YOLO `.pt` model files |
| `MAX_FILE_MB` | `10` | Maximum allowed image file size in MB |
| `IMG_SIZE` | `224` | Image size for YOLO inference |

### Endpoints

#### `GET /health`

Returns model and server status.

**Response:**

```json
{
  "status": "ok",
  "modelLoaded": true,
  "numClasses": 7,
  "classNames": ["melanoma", "chickenpox", "eczema", "hives", "..."],
  "imgSize": [224, 224]
}
```

#### `POST /predict`

Upload a skin disease image and receive a classification result.

**Request:** `multipart/form-data` with an `image` field.

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/image.jpg"
```

**Response:**

```json
{
  "modelLoaded": true,
  "predictedClass": "melanoma",
  "diagnosis": "melanoma",
  "confidence": 0.923,
  "riskLevel": "high",
  "probabilities": {
    "melanoma": 0.923,
    "chickenpox": 0.042,
    "eczema": 0.021,
    "hives": 0.014
  }
}
```

**Error Response (no model):**

```json
{
  "modelLoaded": false,
  "predictedClass": null,
  "diagnosis": null,
  "confidence": null,
  "riskLevel": null,
  "probabilities": null,
  "error": "Model not loaded. Check server logs."
}
```

---

## Risk Level Mapping

| Risk Level | Conditions |
|------------|-----------|
| **HIGH** | Contains "melanoma" in class name |
| **MEDIUM** | Contains "chickenpox", "eczema", or "hives" in class name |
| **LOW** | All other classes |

---

## Dependencies

- **numpy** ≥ 1.24
- **Pillow** ≥ 10.0
- **Flask** — web framework
- **ultralytics** — YOLO model loading and inference

---

## ⚠️ Disclaimer

This tool is intended for **research and educational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified dermatologist for clinical evaluation of skin conditions.
