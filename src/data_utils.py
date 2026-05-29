import pandas as pd
from pathlib import Path

from .config import (
    TRAIN_HISTORY_FILE,
    CLASSIFICATION_REPORT_FILE,
    EVALUATION_SUMMARY_FILE,
)


def load_training_history() -> pd.DataFrame:
    return pd.read_csv(TRAIN_HISTORY_FILE)


def load_classification_report() -> pd.DataFrame:
    return pd.read_csv(CLASSIFICATION_REPORT_FILE)


def load_evaluation_summary() -> pd.DataFrame:
    return pd.read_csv(EVALUATION_SUMMARY_FILE)
