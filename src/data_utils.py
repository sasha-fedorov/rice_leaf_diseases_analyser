from pathlib import Path
import random
from typing import Dict, List

import pandas as pd
import streamlit as st

from .config import (
    CLASS_DISTRIBUTION_FILE,
    CLASSIFICATION_REPORT_FILE,
    EVALUATION_SUMMARY_FILE,
    IMAGE_EXTENSIONS,
    TRAIN_HISTORY_FILE,
    VISUAL_SAMPLE_IMAGES_DIR,
    VISUAL_SAMPLE_LABELS_DIR,
)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_training_history() -> pd.DataFrame:
    return _read_csv(TRAIN_HISTORY_FILE)


@st.cache_data
def load_classification_report() -> pd.DataFrame:
    report = pd.read_csv(CLASSIFICATION_REPORT_FILE, index_col=0)
    return report


@st.cache_data
def load_evaluation_summary() -> pd.DataFrame:
    return _read_csv(EVALUATION_SUMMARY_FILE)


@st.cache_data
def load_class_distribution() -> pd.DataFrame:
    return _read_csv(CLASS_DISTRIBUTION_FILE)


@st.cache_data
def get_sample_images_by_class(
    limit_per_class: int = 1
) -> Dict[str, List[Path]]:
    class_map = {}

    for label_path in VISUAL_SAMPLE_LABELS_DIR.glob("*.txt"):
        if label_path.stat().st_size == 0:
            continue
        with open(label_path, "r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
        if not first_line:
            continue
        parts = first_line.split()
        class_id = parts[0]
        image_name = label_path.stem
        image_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = (
                VISUAL_SAMPLE_IMAGES_DIR / f"{image_name}{ext}"
            )
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        class_map.setdefault(class_id, []).append(image_path)

    sample_map = {}
    for class_id, paths in class_map.items():
        sample_map[class_id] = random.sample(
            paths, min(limit_per_class, len(paths))
        )
    return sample_map


@st.cache_data
def load_class_index_mapping() -> Dict[int, str]:
    from .model_utils import load_class_indices

    return load_class_indices()
