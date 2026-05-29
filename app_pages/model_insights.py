import streamlit as st


def app():
    st.title("Model Insights")
    st.markdown(
        "Review the machine learning model used for rice leaf disease"
        "classification and examine its performance against business"
        "success criteria."
    )

    st.header("Model Overview")
    st.markdown(
        "A MobileNetV2-based classifier was trained for 9-class leaf disease"
        "prediction using transfer learning and class weighting."
    )

    st.header("Performance Metrics")
    st.markdown(
        "The model performance is evaluated using accuracy, precision, recall,"
        "and F1-score for each disease category."
    )

    st.header("Evaluation Summary")
    st.markdown(
        "Key results include validation accuracy, macro F1 score, and areas of"
        "strong and weak performance."
    )
