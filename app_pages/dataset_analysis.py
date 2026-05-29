import streamlit as st


def app():
    st.title("Dataset Analysis")
    st.markdown(
        "Explore the rice leaf dataset, examine class balance, and review "
        "image characteristics that inform model performance."
    )

    st.header("Class Distribution")
    st.markdown(
        "This page will display class imbalance and help identify which "
        "disease categories may need special attention during training."
    )

    st.header("Visual Sample Review")
    st.markdown(
        "Sample images from each class can be shown here to compare healthy "
        "and diseased leaf appearances."
    )

    st.header("Hypothesis Validation")
    st.markdown(
        "The exploratory analysis supports the idea that healthy and diseased "
        "leaves are visually different, and that distinct disease patterns"
        "exist."
    )
