import numpy as np
import streamlit as st
from PIL import Image

from src.data_utils import (
    get_sample_images_by_class,
    load_class_distribution,
    load_class_index_mapping,
)
from src.visualization_utils import plot_class_distribution


def _average_images(paths, size=(256, 256)):
    arrs = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB").resize(size)
            arrs.append(np.array(im, dtype=np.float32))
        except Exception:
            continue
    if not arrs:
        return None
    avg = np.mean(arrs, axis=0).astype(np.uint8)
    return Image.fromarray(avg)


def app():
    st.title("Dataset Analysis & Exploratory Data Analysis")
    st.markdown(
        "Explore the rice leaf dataset, examine class balance, and review "
        "image characteristics that inform model performance."
    )

    # Load data
    distribution = load_class_distribution()
    class_mapping = load_class_index_mapping()

    # Section 1: Class Distribution
    st.header("1. Class Distribution")
    st.write(
        "The dataset includes 9 classes with significant imbalance between "
        "common diseases and less frequent categories. This imbalance affects "
        "model training and performance."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.pyplot(plot_class_distribution(distribution))
    with col2:
        st.subheader("Summary Statistics")
        total = distribution["Total Images"].sum()
        st.metric("Total Images", int(total))
        st.metric("Number of Classes", len(distribution))
        max_class = distribution.loc[distribution["Total Images"].idxmax()]
        st.metric(
            "Largest Class",
            f"{max_class['Class']}({int(max_class['Total Images'])})"
        )

    st.subheader("Class Breakdown")
    dist_display = distribution.copy()

    train_imgs = dist_display["Train Images"]
    dist_display["Train %"] = (train_imgs / train_imgs.sum() * 100).round(1)
    val_imgs = dist_display["Validation Images"]
    dist_display["Val %"] = (val_imgs / val_imgs.sum() * 100).round(1)

    st.dataframe(dist_display.set_index("Class"), width="stretch")

    # Section 2: Visual Sample Review
    st.header("2. Visual Sample Review")
    st.write(
        "Representative images from each class help validate whether healthy "
        "leaves and disease categories are visually distinct."
    )

    samples = get_sample_images_by_class(limit_per_class=3)

    # Visual Examples: comparison tools
    st.subheader("Visual Examples")
    compare_mode = st.radio(
        "Comparison type:",
        ["Healthy vs Disease",
         "Disease vs Disease"]
    )

    name_to_id = {v: str(k) for k, v in class_mapping.items()}

    if compare_mode == "Healthy vs Disease":
        disease_choice = st.selectbox(
            "Choose disease to compare with Healthy:",
            options=[
                n for n in sorted(class_mapping.values())
                if n.lower() != "healthy"
            ],
        )
        healthy_id = name_to_id.get("Healthy") or name_to_id.get("healthy")
        disease_id = name_to_id.get(disease_choice)

        healthy_paths = samples.get(healthy_id, [])
        disease_paths = samples.get(disease_id, [])

        colL, colR = st.columns(2)
        with colL:
            st.markdown("**Healthy samples**")
            if healthy_paths:
                for p in healthy_paths:
                    st.image(str(p), width="stretch")
            else:
                st.info("No healthy samples found.")
            avg_h = _average_images(healthy_paths)
            if avg_h is not None:
                st.markdown("Average healthy image")
                st.image(avg_h, width="stretch")
        with colR:
            st.markdown(f"**{disease_choice} samples**")
            if disease_paths:
                for p in disease_paths:
                    st.image(str(p), width="stretch")
            else:
                st.info(f"No samples found for {disease_choice}.")
            avg_d = _average_images(disease_paths)
            if avg_d is not None:
                st.markdown(f"Average {disease_choice} image")
                st.image(avg_d, width="stretch")

    else:
        disease_options = sorted(class_mapping.values())
        pair = st.multiselect(
            "Select two diseases to compare:",
            options=disease_options,
            default=disease_options[:2]
        )
        if len(pair) != 2:
            st.info("Select exactly two classes to compare.")
        else:
            id_a = name_to_id.get(pair[0])
            id_b = name_to_id.get(pair[1])
            paths_a = samples.get(id_a, [])
            paths_b = samples.get(id_b, [])

            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**{pair[0]} samples**")
                if paths_a:
                    for p in paths_a:
                        st.image(str(p), width="stretch")
                else:
                    st.info(f"No samples for {pair[0]}.")
                avg_a = _average_images(paths_a)
                if avg_a is not None:
                    st.markdown(f"Average {pair[0]} image")
                    st.image(avg_a, width="stretch")
            with colB:
                st.markdown(f"**{pair[1]} samples**")
                if paths_b:
                    for p in paths_b:
                        st.image(str(p), width="stretch")
                else:
                    st.info(f"No samples for {pair[1]}.")
                avg_b = _average_images(paths_b)
                if avg_b is not None:
                    st.markdown(f"Average {pair[1]} image")
                    st.image(avg_b, width="stretch")

    # Section 3: Dataset Insights
    st.header("3. Dataset Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        healthy_df = distribution[distribution["Class"] == "Healthy"]
        healthy_count = healthy_df["Total Images"].sum()
        disease_count = total - healthy_count
        st.metric("Healthy Images", int(healthy_count))
    with col2:
        st.metric("Disease Images", int(disease_count))
    with col3:
        total_images = distribution["Total Images"]
        imbalance_ratio = max(total_images) / min(total_images)
        st.metric("Max Class Imbalance", f"{imbalance_ratio:.1f}x")

    st.markdown(
        "**Key Observation:** Brown Spot and Leaf Smut are over-represented "
        "with 2,000 images each, while other disease classes have only "
        "~350-450 images. This imbalance is addressed during model training "
        "through class weighting."
    )

    # Section 4: Hypothesis Validation
    st.header("4. Hypothesis Validation from EDA")

    st.subheader(
        "Hypothesis 1: Healthy & Diseased Leaves Have Distinct Visual Patterns"
    )
    st.markdown(
        "**Status: VALIDATED**\n\n"
        "The visual samples show clear differences between healthy leaves and "
        "diseased leaves. Healthy leaves appear uniform in color, while "
        "diseased leaves show distinct lesions, discoloration, or texture "
        "changes depending on the disease type."
    )

    st.subheader("Hypothesis 2: Different Diseases Can Be Distinguished")
    st.markdown(
        "**Status: VALIDATED**\n\n"
        "Each disease exhibits unique visual characteristics:\n"
        "- **Leaf Blast:** Gray spots with concentric rings\n"
        "- **Brown Spot:** Brown lesions with yellowish halos\n"
        "- **Bacterial Leaf Blight:** Linear yellowing along veins\n"
        "- **Leaf Scald:** Drying of leaf edges\n"
        "- Other diseases show distinct patterns suitable for classification."
    )

    st.subheader("Hypothesis 3: Class Imbalance Influences Model Performance")
    st.markdown(
        "**Status: CONFIRMED**\n\n"
        "The dataset shows severe class imbalance (Brown Spot and Leaf Smut: "
        "2,000 images; Neck Blast: 453 images). This imbalance requires "
        "mitigation strategies like class weighting and data augmentation "
        "during model training to ensure minority classes are learned "
        "effectively."
    )
