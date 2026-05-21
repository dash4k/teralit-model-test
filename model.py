"""
Skin Cancer Classification using SVM (Support Vector Machine)
Dataset: HAM10000 (Skin Cancer MNIST) from Kaggle
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Dataset classes (7 lesion types):
    - akiec: Actinic Keratoses and Intraepithelial Carcinoma
    - bcc  : Basal Cell Carcinoma
    - bkl  : Benign Keratosis-like Lesions
    - df   : Dermatofibroma
    - mel  : Melanoma
    - nv   : Melanocytic Nevi
    - vasc : Vascular Lesions

Expected directory structure after downloading:
    data/
    ├── HAM10000_metadata.csv
    ├── HAM10000_images_part_1/   (*.jpg)
    └── HAM10000_images_part_2/   (*.jpg)
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR = Path("data")                        # Root data directory
IMG_DIRS = [                                   # Folders containing images
    DATA_DIR / "HAM10000_images_part_1",
    DATA_DIR / "HAM10000_images_part_2",
]
METADATA_CSV = DATA_DIR / "HAM10000_metadata.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = (64, 64)          # Resize images to this (H, W)
N_COMPONENTS = 100           # PCA components (dimensionality reduction)
TEST_SIZE = 0.2              # 80/20 train-test split
RANDOM_STATE = 42
USE_GRID_SEARCH = False      # Set True for hyperparameter tuning (slower)
MAX_SAMPLES = None           # Set an int (e.g. 5000) to cap dataset for quick runs

CLASS_LABELS = {
    "akiec": "Actinic Keratoses",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}

# ─────────────────────────────────────────────
# STEP 1 – Load metadata
# ─────────────────────────────────────────────
def load_metadata(csv_path: Path) -> pd.DataFrame:
    print(f"[1/6] Loading metadata from {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"      Total records : {len(df):,}")
    print(f"      Class distribution:\n{df['dx'].value_counts().to_string()}\n")
    return df


# ─────────────────────────────────────────────
# STEP 2 – Build image-path lookup
# ─────────────────────────────────────────────
def build_image_index(img_dirs: list[Path]) -> dict[str, Path]:
    """Return {image_id: full_path} from all image directories."""
    index: dict[str, Path] = {}
    for d in img_dirs:
        if not d.exists():
            print(f"      WARNING: image directory not found → {d}")
            continue
        for p in d.glob("*.jpg"):
            index[p.stem] = p
    print(f"      Indexed {len(index):,} image files.")
    return index


# ─────────────────────────────────────────────
# STEP 3 – Load & preprocess images
# ─────────────────────────────────────────────
def load_images(
    df: pd.DataFrame,
    img_index: dict[str, Path],
    img_size: tuple[int, int],
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load images, resize, flatten into feature vectors.
    Returns X (n_samples, H*W*3) and y (n_samples,) label strings.
    """
    print(f"[2/6] Loading images (size={img_size}) …")

    if max_samples:
        df = df.groupby("dx", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max_samples // len(df["dx"].unique())),
                               random_state=RANDOM_STATE)
        ).reset_index(drop=True)
        print(f"      Capped to {len(df):,} samples for speed.")

    X, y = [], []
    missing = 0
    for _, row in df.iterrows():
        img_id = row["image_id"]
        label  = row["dx"]
        path   = img_index.get(img_id)

        if path is None:
            missing += 1
            continue

        img = Image.open(path).convert("RGB").resize(img_size)
        X.append(np.array(img, dtype=np.float32).flatten() / 255.0)
        y.append(label)

    if missing:
        print(f"      Skipped {missing} missing images.")

    X = np.array(X)
    y = np.array(y)
    print(f"      Feature matrix : {X.shape}  (samples × pixels)\n")
    return X, y


# ─────────────────────────────────────────────
# STEP 4 – Build & train the SVM pipeline
# ─────────────────────────────────────────────
def build_pipeline(n_components: int, class_weights: dict) -> Pipeline:
    """StandardScaler → PCA → SVM pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ("svm",    SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight=class_weights,
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int,
    class_weights: dict,
    use_grid_search: bool = False,
) -> Pipeline:
    print("[4/6] Training SVM …")
    pipeline = build_pipeline(n_components, class_weights)

    if use_grid_search:
        print("      Running GridSearchCV (this may take a while) …")
        param_grid = {
            "svm__C":     [1, 10, 100],
            "svm__gamma": ["scale", "auto", 0.01],
            "svm__kernel": ["rbf", "poly"],
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        search = GridSearchCV(
            pipeline, param_grid, cv=cv,
            scoring="balanced_accuracy", n_jobs=-1, verbose=2,
        )
        search.fit(X_train, y_train)
        print(f"      Best params : {search.best_params_}")
        print(f"      Best CV score: {search.best_score_:.4f}\n")
        return search.best_estimator_

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"      Training finished in {time.time() - t0:.1f}s\n")
    return pipeline


# ─────────────────────────────────────────────
# STEP 5 – Evaluate
# ─────────────────────────────────────────────
def evaluate(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: Path,
) -> None:
    print("[5/6] Evaluating …")
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n      Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)\n")

    class_names = label_encoder.classes_

    # ── Classification report ──────────────────
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("      Classification Report:")
    print(report)

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report)
    print(f"      Saved → {report_path}")

    # ── ROC-AUC (macro, OvR) ──────────────────
    try:
        auc = roc_auc_score(
            label_encoder.transform(y_test),
            y_proba,
            multi_class="ovr",
            average="macro",
        )
        print(f"\n      ROC-AUC (macro OvR) : {auc:.4f}")
    except Exception as e:
        print(f"      ROC-AUC skipped: {e}")

    # ── Confusion matrix plot ─────────────────
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title("Confusion Matrix – SVM (HAM10000)", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"      Saved → {cm_path}\n")


# ─────────────────────────────────────────────
# STEP 6 – Save model
# ─────────────────────────────────────────────
def save_model(pipeline: Pipeline, label_encoder: LabelEncoder, output_dir: Path) -> None:
    model_path = output_dir / "svm_skin_cancer.pkl"
    le_path    = output_dir / "label_encoder.pkl"
    joblib.dump(pipeline,      model_path)
    joblib.dump(label_encoder, le_path)
    print(f"[6/6] Model saved  → {model_path}")
    print(f"      Encoder saved → {le_path}\n")


# ─────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────
def predict_single_image(
    img_path: str,
    model_path: str  = "outputs/svm_skin_cancer.pkl",
    le_path: str     = "outputs/label_encoder.pkl",
    img_size: tuple  = IMG_SIZE,
) -> dict:
    """
    Predict the lesion class for a single image.

    Returns:
        {
          "predicted_class": "nv",
          "label": "Melanocytic Nevi",
          "probabilities": {"akiec": 0.01, "bcc": 0.02, ...}
        }
    """
    pipeline = joblib.load(model_path)
    le       = joblib.load(le_path)

    img = Image.open(img_path).convert("RGB").resize(img_size)
    x   = np.array(img, dtype=np.float32).flatten() / 255.0

    pred_class = pipeline.predict([x])[0]
    proba      = pipeline.predict_proba([x])[0]

    return {
        "predicted_class": pred_class,
        "label": CLASS_LABELS.get(pred_class, pred_class),
        "probabilities": dict(zip(le.classes_, proba.round(4))),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  Skin Cancer Classification – SVM  (HAM10000)")
    print("=" * 60, "\n")

    # 1 – Metadata
    df = load_metadata(METADATA_CSV)

    # 2 – Image index
    print("[1.5/6] Indexing image files …")
    img_index = build_image_index(IMG_DIRS)

    # 3 – Load images
    X, y = load_images(df, img_index, IMG_SIZE, max_samples=MAX_SAMPLES)

    # 3b – Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"[3/6] Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

    # 3c – Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
    )
    print(f"      Train samples : {len(X_train):,}")
    print(f"      Test  samples : {len(X_test):,}\n")

    # Decode y_test back to string for readable reports
    y_test_str  = le.inverse_transform(y_test)
    y_train_str = le.inverse_transform(y_train)

    # 3d – Class weights (handle imbalance)
    # Keys must be strings to match y_train_str passed to the SVM
    classes_str = np.unique(y_train_str)
    weights     = compute_class_weight("balanced", classes=classes_str, y=y_train_str)
    class_weights = {cls: float(w) for cls, w in zip(classes_str, weights)}

    # 4 – Train
    pipeline = train(X_train, y_train_str, N_COMPONENTS, class_weights, USE_GRID_SEARCH)

    # 5 – Evaluate
    evaluate(pipeline, X_test, y_test_str, le, OUTPUT_DIR)

    # 6 – Save
    save_model(pipeline, le, OUTPUT_DIR)

    print("Done! All outputs are in the 'outputs/' directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()