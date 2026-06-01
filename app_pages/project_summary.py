import streamlit as st

from src.data_utils import load_class_distribution, load_evaluation_summary


def app():
    st.title("Rice Leaf Diseases Analyser")
    st.markdown(
        "A farmer-facing dashboard that uses image-based machine learning to "
        "detect rice leaf diseases and provide actionable insights for crop "
        "health."
    )

    summary = load_evaluation_summary().set_index("metric")
    class_dist = load_class_distribution()
    total_images = int(class_dist["Total Images"].sum())
    num_classes = len(class_dist)

    st.subheader("Project Overview")
    st.markdown(
        "This project applies a CNN model to a rice leaf dataset in YOLO "
        "format for a multi-class classification task. The goal is to "
        "identify whether a leaf is healthy or infected, and if infected, "
        "determine the specific disease type from 8 categories."
    )

    st.subheader("Business Context")
    st.markdown(
        "Rice farmers need a fast, reliable diagnostic tool to reduce crop "
        "losses from leaf diseases and improve treatment decisions. "
        "This dashboard translates model predictions into clear, actionable "
        "information."
    )

    st.header("Key Results")
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
        "- Manual inspection is unreliable and slow for large rice fields.\n"
        "- Farmers need consistent disease identification for targeted "
        "treatment.\n"
        "- Automated diagnostics can reduce crop loss and support advisory "
        "services."
    )

    st.header("Project Hypotheses")
    st.markdown(
        "1. Healthy and diseased leaves show distinct visual patterns.\n"
        "2. Different rice diseases can be distinguished from leaf images.\n"
        "3. Class imbalance affects model performance and requires mitigation."
    )

    st.header("Business Requirements")
    st.markdown(
        "- Requirement 1: Provide visual evidence to differentiate healthy "
        "leaves from diseased leaves.\n"
        "- Requirement 2: Automate accurate disease classification and supply "
        "confidence estimates for each prediction."
    )

    st.header("Dashboard Navigation")
    st.markdown(
        "- **Project Summary:** Business context, dataset overview, "
        "and model goals.\n"
        "- **Dataset Analysis:** Class balance, sample visuals, "
        "and EDA insights.\n"
        "- **Model Insights:** Performance metrics, training curves, "
        "and confusion matrix.\n"
        "- **Disease Predictor:** Upload images and receive disease "
        "prediction with confidence.\n"
        "- **Business Conclusions:** Final recommendations and model "
        "applicability."
    )

    st.header("How the App Delivers Value")
    st.markdown(
        "This application integrates dataset exploration, model evaluation, "
        "and an interactive prediction experience so stakeholders can trust "
        "the ML solution and use it for informed farming decisions."
    )
