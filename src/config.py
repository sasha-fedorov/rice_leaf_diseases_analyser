from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "outputs" / "models"
EVAL_DIR = ROOT_DIR / "outputs" / "evaluation"
DATA_DIR = ROOT_DIR / "inputs" / "datasets"

MODEL_FILE = MODEL_DIR / "rice_leaf_disease_classifier.keras"
CLASS_INDEX_FILE = MODEL_DIR / "class_indices.json"
TRAIN_HISTORY_FILE = MODEL_DIR / "training_history.csv"
CLASSIFICATION_REPORT_FILE = EVAL_DIR / "classification_report.csv"
CONFUSION_MATRIX_FILE = EVAL_DIR / "confusion_matrix.png"
EVALUATION_SUMMARY_FILE = EVAL_DIR / "evaluation_summary.csv"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
