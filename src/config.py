from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODEL_DIR = OUTPUTS_DIR / "models"
EVAL_DIR = OUTPUTS_DIR / "evaluation"
DATA_DIR = ROOT_DIR / "inputs" / "datasets"
RAW_DATA_DIR = DATA_DIR / "raw" / "rice"
RAW_LABELS_DIR = RAW_DATA_DIR / "labels"
SAMPLES_DIR = OUTPUTS_DIR / "datasets/eda/samples"
VISUAL_SAMPLE_IMAGES_DIR = SAMPLES_DIR / "images"
VISUAL_SAMPLE_LABELS_DIR = SAMPLES_DIR / "labels"

MODEL_FILE = MODEL_DIR / "rice_leaf_disease_classifier.keras"
CLASS_INDEX_FILE = MODEL_DIR / "class_indices.json"
TRAIN_HISTORY_FILE = MODEL_DIR / "training_history.csv"
CLASSIFICATION_REPORT_FILE = EVAL_DIR / "classification_report.csv"
CONFUSION_MATRIX_FILE = EVAL_DIR / "confusion_matrix.png"
EVALUATION_SUMMARY_FILE = EVAL_DIR / "evaluation_summary.csv"
CLASS_DISTRIBUTION_FILE = (
    ROOT_DIR / "outputs" / "datasets" / "eda" / "class_distribution.csv"
)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
