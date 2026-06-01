import streamlit as st

from src.data_utils import (
    load_class_distribution,
    get_sample_images_by_class,
    load_class_index_mapping,
)
from src.visualization_utils import plot_class_distribution


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

    st.dataframe(dist_display.set_index("Class"), use_container_width=True)

    # Section 2: Visual Sample Review
    st.header("2. Visual Sample Review")
    st.write(
        "Representative images from each class help validate whether healthy "
        "leaves and disease categories are visually distinct."
    )

    class_filter = st.multiselect(
        "Filter by class (select to show samples):",
        options=sorted(class_mapping.values()),
        default=list(class_mapping.values())
    )

    if class_filter:
        samples = get_sample_images_by_class(limit_per_class=2)
        class_mapping_inv = {v: k for k, v in class_mapping.items()}

        sample_items = []
        for class_name in class_filter:
            class_id = str(class_mapping_inv.get(class_name, ""))
            if class_id in samples:
                for image_path in samples[class_id]:
                    sample_items.append((class_name, image_path))

        if sample_items:
            for idx in range(0, len(sample_items), 3):
                cols = st.columns(3)
                for column, item in zip(cols, sample_items[idx: idx + 3]):
                    class_name, image_path = item
                    with column:
                        st.image(
                            str(image_path),
                            caption=class_name,
                            use_container_width=True
                        )
        else:
            st.warning("No sample images found for the selected classes.")
    else:
        st.info("Select classes above to view sample images.")

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
        "mitigation strategies like class weighting  and data augmentation "
        "during model training to ensure minority classes are learned "
        "effectively."
    )
