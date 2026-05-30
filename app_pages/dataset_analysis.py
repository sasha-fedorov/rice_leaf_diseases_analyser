import streamlit as st

from src.data_utils import (
    load_class_distribution,
    get_sample_images_by_class,
    load_class_index_mapping,
)
from src.visualization_utils import plot_class_distribution


def app():
    st.title("Dataset Analysis")
    st.markdown(
        "Explore the rice leaf dataset, examine class balance, and review "
        "image characteristics that inform model performance."
    )

    distribution = load_class_distribution()
    st.subheader("Class Distribution")
    st.write(
        "The dataset includes 9 classes with significant imbalance between"
        "common diseases and less frequent categories."
    )
    st.pyplot(plot_class_distribution(distribution))
    st.dataframe(distribution.set_index("Class"))

    st.subheader("Visual Sample Review")
    st.write(
        "Representative images from each class help validate whether healthy"
        "leaves and disease categories are visually distinct."
    )

    samples = get_sample_images_by_class(limit_per_class=1)
    class_mapping = load_class_index_mapping()
    sample_items = []
    for class_id, image_paths in samples.items():
        class_name = class_mapping.get(int(class_id), f"Class {class_id}")
        sample_items.append((class_name, image_paths[0]))

    if sample_items:
        for idx in range(0, len(sample_items), 3):
            cols = st.columns(3)
            for column, item in zip(cols, sample_items[idx: idx + 3]):
                class_name, image_path = item
                column.image(str(image_path),
                             caption=class_name,
                             use_container_width=True)
    else:
        st.warning("No sample images were found in the dataset directory.")

    st.subheader("Hypothesis Validation")
    st.markdown(
        "- Healthy and diseased leaves show distinct visual patterns.\n"
        "- Different diseases can be distinguished from leaf images.\n"
        "- Class imbalance is present and may impact model performance."
    )
