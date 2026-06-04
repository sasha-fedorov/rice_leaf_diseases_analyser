import json
from typing import Dict

import numpy as np
from PIL import Image
from keras.models import load_model

from .config import MODEL_FILE, CLASS_INDEX_FILE


def load_disease_model():
    return load_model(MODEL_FILE)


def load_class_indices() -> Dict[int, str]:
    with open(CLASS_INDEX_FILE, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(k): v for k, v in raw.items()}


def prepare_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)


def _get_model_input_size(model):
    input_shape = getattr(model, "input_shape", None)
    if input_shape is None:
        return (224, 224)
    if isinstance(input_shape, tuple) and len(input_shape) >= 3:
        return tuple(input_shape[1:3])
    if isinstance(input_shape, list) and len(input_shape) > 0:
        shape = input_shape[0]
        if isinstance(shape, tuple) and len(shape) >= 3:
            return tuple(shape[1:3])
    return (224, 224)


def predict_image(model, class_indices: Dict[int, str], image: Image.Image):
    target_size = _get_model_input_size(model)
    img = prepare_image(image, target_size=target_size)
    prediction = model.predict(img)
    probabilities = prediction[0].tolist()
    best_idx = int(np.argmax(prediction[0]))
    return {
        "class_id": best_idx,
        "class_name": class_indices[best_idx],
        "confidence": float(prediction[0][best_idx]),
        "probabilities": probabilities,
    }
