import json
from typing import Dict, List
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

from .config import MODEL_FILE, CLASS_INDEX_FILE, IMAGE_EXTENSIONS


def load_disease_model():
    return load_model(MODEL_FILE)


def load_class_indices() -> Dict[int, str]:
    with open(CLASS_INDEX_FILE, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(k): v for k, v in raw.items()}


def prepare_image(image: Image.Image, target_size=(256, 256)) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_image(model, class_indices: Dict[int, str], image: Image.Image):
    img = prepare_image(image)
    prediction = model.predict(img)
    probabilities = prediction[0].tolist()
    best_idx = int(np.argmax(prediction[0]))
    return {
        "class_id": best_idx,
        "class_name": class_indices[best_idx],
        "confidence": float(prediction[0][best_idx]),
        "probabilities": probabilities,
    }
