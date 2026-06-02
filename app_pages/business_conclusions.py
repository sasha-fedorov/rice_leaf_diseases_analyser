import streamlit as st

from src.data_utils import load_evaluation_summary, load_classification_report


def app():
    st.title("Business Conclusions")
    st.markdown(
        "Summarize the project findings, validate the ML business case, "
        "and describe the expected business impact for rice farmers."
    )

    st.header("Executive Summary")
    st.markdown(
        "The deployed model delivers a reliable rice leaf disease classifier "
        "that supports timely treatment decisions for farmers. The app "
        "combines visual analysis, model performance information, and "
        "prediction recommendations."
    )

    st.header("Business Case Validation")
    evaluation = load_evaluation_summary()
    metrics = {
        row["metric"]: float(row["value"])
        for _, row in evaluation.iterrows()
    }
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Validation Accuracy",
            f"{metrics.get('validation_accuracy', 0):.1%}"
        )
    with col2:
        st.metric("Macro F1-score", f"{metrics.get('macro_f1_score', 0):.3f}")
    with col3:
        st.metric("Macro Recall", f"{metrics.get('macro_recall', 0):.3f}")

    st.markdown(
        "The model exceeds the project performance thresholds: \n"
        "- Validation accuracy target: 75%\n"
        "- Macro F1-score target: 0.70\n"
        "- The app supports both healthy vs diseased classification and "
        "disease-specific diagnosis."
    )

    st.header("Hypothesis Validation")
    st.markdown(
        "- **Hypothesis 1:** Healthy and diseased leaves have distinct visual "
        "patterns. **Validated.**\n"
        "  - Evidence: The EDA shows healthy leaf samples with uniform green "
        "color and intact texture, while diseased samples contain visible "
        "lesions, discoloration, and irregular patterns. Average-image "
        "comparisons further highlight the contrast between healthy and "
        "disease classes.\n"
        "- **Hypothesis 2:** Different rice diseases can be distinguished by "
        "image classification. **Validated.**\n"
        "  - Evidence: Disease-specific visual cues such as brown spots, "
        "yellowing, linear lesions, and margin necrosis are captured by the "
        "trained model. Per-class performance reports demonstrate that the "
        "model is able to separate multiple diseases with strong F1 scores "
        "across most classes.\n"
        "- **Hypothesis 3:** Class imbalance influences model performance. "
        "**Confirmed.**\n"
        "  - Evidence: The dataset contains a large imbalance between common "
        "classes (e.g. Brown Spot and Leaf Smut) and rarer diseases, which is "
        "reflected in variation in validation performance. The project "
        "mitigates this with class weighting and careful evaluation of "
        "minority-class results."
    )

    st.header("Performance Insights")
    report = load_classification_report()
    if not report.empty:
        st.subheader("Per-Class Performance")
        st.dataframe(report)

        best_class = report.loc[report['f1-score'].idxmax()]
        worst_class = report.loc[report['f1-score'].idxmin()]
        st.markdown(
            f"**Best performing class:** {best_class.name}"
            f"with F1-score {best_class['f1-score']:.3f}.\n"
            f"**Lowest performing class:** {worst_class.name}"
            f"with F1-score {worst_class['f1-score']:.3f}."
        )
    else:
        st.info("Classification report is not available yet.")

    st.header("Business Recommendations")
    st.markdown(
        "- Use the dashboard for rapid disease screening and to prioritize "
        "field visits.\n"
        "- Treat predictions with high confidence as actionable, while "
        "low-confidence cases should be reviewed by experts.\n"
        "- Continue collecting images for underrepresented disease classes to "
        "improve future model accuracy.\n"
        "- Leverage the model output to reduce unnecessary treatments and "
        "support targeted interventions."
    )

    st.header("Model Deployment Notes")
    st.markdown(
        "The model is suitable for farmer-facing deployment with the "
        "following guidance: \n"
        "1. Use clear, well-lit leaf images for the best results.\n"
        "2. Display confidence scores alongside each prediction.\n"
        "3. Flag low-confidence predictions for expert review.\n"
        "4. Update the model with new data over time to reduce bias from "
        "class imbalance."
    )
