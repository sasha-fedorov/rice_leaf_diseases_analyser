import streamlit as st

from src.data_utils import load_class_distribution, load_evaluation_summary


def app():
    st.title("Rice Leaf Diseases Analyser")
    st.markdown(
        "This dashboard presents a machine learning solution for rice farmers"
        "to identify leaf diseases and support treatment decisions."
    )

    summary = load_evaluation_summary().set_index("metric")
    class_dist = load_class_distribution()
    total_images = int(class_dist["Total Images"].sum())
    num_classes = len(class_dist)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset at a Glance")
        st.metric("Total Classes", num_classes)
        st.metric("Total Images", total_images)
        st.markdown(
            "- 9 classes: Healthy + 8 rice disease types\n"
            "- YOLO-format dataset used for single-label classification\n"
            "- Train and validation splits available"
        )
    with col2:
        st.subheader("Model Performance")
        st.metric("Validation Accuracy",
                  f"{float(summary.loc['validation_accuracy', 'value']):.1%}")
        st.metric("Macro F1 Score",
                  f"{float(summary.loc['macro_f1_score', 'value']):.3f}")
        st.markdown(
            "- The model exceeds the target accuracy of 75%\n"
            "- Macro F1 score is above the 0.70 success threshold"
        )

    st.header("Business Challenge")
    st.markdown(
        "- Manual disease detection is inconsistent and time-consuming.\n"
        "- Farmers need fast, reliable plant health insights.\n"
        "- Accurate disease classification enables targeted treatment."
    )

    st.header("Project Hypotheses")
    st.markdown(
        "1. Healthy and diseased leaves show distinct visual patterns.\n"
        "2. Different diseases can be distinguished from leaf images.\n"
        "3. Class imbalance influences model performance."
    )

    st.header("Business Requirements")
    st.markdown(
        "- Requirement 1: Visual evidence and disease analysis.\n"
        "- Requirement 2: Automated disease classification with confidence"
        "scoring."
    )

    st.header("How the App Delivers Value")
    st.markdown(
        "This application combines data exploration, model insights, and an"
        "interactive predictor to help farmers identify leaf diseases quickly"
        "and with confidence."
    )
