import streamlit as st


def app():
    st.title("Rice Leaf Diseases Analyser")
    st.markdown(
        "This dashboard presents a machine learning solution for rice farmers"
        "to identify leaf diseases and support treatment decisions."
    )

    st.header("Business Challenge")
    st.markdown(
        "- Manual disease detection is inconsistent and time-consuming.\n"
        "- Farmers need fast, reliable plant health insights.\n"
        "- Accurate disease classification enables targeted treatment."
    )

    st.header("Dataset Overview")
    st.markdown(
        "The model is trained on a rice leaf dataset with 9 classes: Healthy"
        "plus 8 disease categories."
    )
    st.markdown(
        "The dataset is stored in YOLO format, but used here as a single-label"
        "classification task."
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
