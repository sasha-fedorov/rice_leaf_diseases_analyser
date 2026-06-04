from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from src.config import IMAGE_EXTENSIONS, RAW_LABELS_DIR
from src.model_utils import (
    load_class_indices,
    load_disease_model,
    predict_image,
)


@st.cache_resource
def get_model():
    return load_disease_model()


def find_label_for_uploaded(filename: str):
    name = Path(filename).stem
    for split in ("train", "val"):
        p = RAW_LABELS_DIR / split / f"{name}.txt"
        if p.exists():
            return p
    return None


def draw_yolo_bbox(image: Image.Image, label_path: Path) -> Image.Image:
    try:
        with open(label_path, "r", encoding="utf-8") as fh:
            line = fh.readline().strip()
        if not line:
            return image
        parts = line.split()
        if len(parts) < 5:
            return image
        # YOLO format: class x_center y_center width height (normalized)
        _, xc, yc, w, h = parts[0:5]
        xc = float(xc)
        yc = float(yc)
        w = float(w)
        h = float(h)
        img_w, img_h = image.size
        x_c = xc * img_w
        y_c = yc * img_h
        bw = w * img_w
        bh = h * img_h
        left = int(x_c - bw / 2)
        top = int(y_c - bh / 2)
        right = int(x_c + bw / 2)
        bottom = int(y_c + bh / 2)
        draw = ImageDraw.Draw(image)
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        return image
    except Exception:
        return image


def app():
    st.title("Disease Predictor")
    st.markdown(
        "Upload one or more rice leaf images to get disease classifications "
        "with confidence scores."
    )

    uploaded_files = st.file_uploader(
        "Choose image files",
        type=[ext.strip(".") for ext in IMAGE_EXTENSIONS],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Upload one or more images to get predictions.")
        return

    model = get_model()
    class_indices = load_class_indices()

    results = []

    for uploaded in uploaded_files:
        try:
            img = Image.open(uploaded).convert("RGB")
        except Exception:
            st.warning(f"Failed to open {uploaded.name}")
            continue

        pred = predict_image(model, class_indices, img)

        # Prepare top-3 predictions
        probs = pred.get("probabilities", [])
        top3 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
        top3_list = []
        for idx, p in top3:
            cname = class_indices.get(idx, str(idx))
            top3_list.append({"class": cname, "confidence": float(p)})

        # Draw bbox if label file exists for this image name
        label_path = find_label_for_uploaded(uploaded.name)
        display_img = img.copy()
        if label_path is not None:
            display_img = draw_yolo_bbox(display_img, label_path)

        # Display results
        st.subheader(uploaded.name)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(display_img, width="stretch")
        with col2:
            st.markdown(f"**Prediction:** {pred['class_name']}\n")
            st.markdown(f"**Confidence:** {pred['confidence']:.2%}\n")
            st.table(pd.DataFrame(top3_list))

        results.append({
            "filename": uploaded.name,
            "prediction": pred["class_name"],
            "confidence": float(pred["confidence"]),
        })

    if results:
        df = pd.DataFrame(results)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions (CSV)",
            data=csv, file_name="predictions.csv",
            mime="text/csv"
        )
