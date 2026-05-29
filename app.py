import streamlit as st

from app_pages import (
    project_summary,
    dataset_analysis,
    model_insights,
    disease_predictor,
    business_conclusions,
)

PAGES = {
    "1. Project Summary": project_summary,
    "2. Dataset Analysis": dataset_analysis,
    "3. Model Insights": model_insights,
    "4. Disease Predictor": disease_predictor,
    "5. Business Conclusions": business_conclusions,
}

st.set_page_config(
    page_title="Rice Leaf Diseases Analyser",
    page_icon="🌾",
    layout="wide",
)

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Choose a page:", list(PAGES.keys()))
page = PAGES[selection]
page.app()
