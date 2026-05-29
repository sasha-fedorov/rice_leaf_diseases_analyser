import streamlit as st


def app():
    st.title("Business Conclusions")
    st.markdown(
        "Summarize the project findings, validate the ML business case,"
        "and describe the expected business impact for rice farmers."
    )

    st.header("Summary of Results")
    st.markdown(
        "The model meets the project success metrics and provides "
        "a strong foundation for an agricultural decision support tool."
    )

    st.header("Recommendations")
    st.markdown(
        "The dashboard can be used to support disease treatment "
        "decisions and identify cases requiring expert review."
    )
