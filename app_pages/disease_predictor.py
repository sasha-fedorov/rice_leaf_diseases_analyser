import streamlit as st


def app():
    st.title("Disease Predictor")
    st.markdown(
        "Upload a rice leaf image to get a disease classification and"
        "confidence estimate."
    )

    st.header("Image Upload")
    st.markdown(
        "This page will allow users to upload leaf images and view model"
        "predictions for disease type and confidence."
    )

    st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    st.header("Prediction Result")
    st.markdown(
        "After uploading an image, the model will display the predicted"
        "disease class, confidence score, and recommendations."
    )
