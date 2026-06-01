import streamlit as st

from src.data_utils import (
    load_training_history,
    load_classification_report,
    load_evaluation_summary,
)
from src.visualization_utils import (
    plot_training_curves,
    plot_class_report,
    plot_confusion_matrix_from_file,
)
from src.config import CONFUSION_MATRIX_FILE


def app():
    st.title("Model Insights")
    st.markdown(
        "Review the machine learning model used for rice leaf disease "
        "classification and examine its performance against business "
        "success criteria."
    )

    # Load data
    try:
        eval_summary = load_evaluation_summary()
    except Exception:
        eval_summary = None

    try:
        history = load_training_history()
    except Exception:
        history = None

    try:
        class_report = load_classification_report()
    except Exception:
        class_report = None

    st.header("Evaluation Summary")
    if eval_summary is not None:
        # display key metrics as cards
        metrics = (
            {r["metric"]: r["value"] for _, r in eval_summary.iterrows()}
            if hasattr(eval_summary, "iterrows")
            else {}
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'validation_accuracy' in metrics:
                st.metric(
                    "Validation Accuracy",
                    f"{float(metrics['validation_accuracy']):.1%}")
        with col2:
            if 'macro_f1_score' in metrics:
                st.metric(
                    "Macro F1-score",
                    f"{float(metrics['macro_f1_score']):.3f}")
        with col3:
            if 'macro_recall' in metrics:
                st.metric(
                    "Macro Recall",
                    f"{float(metrics['macro_recall']):.3f}")
    else:
        st.warning("Evaluation summary file not available.")

    st.header("Training Curves")
    if history is not None:
        try:
            fig = plot_training_curves(history)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Failed to render training curves: {e}")
    else:
        st.info("Training history not found. Skipping training curves.")

    st.header("Confusion Matrix")
    try:
        if CONFUSION_MATRIX_FILE.exists():
            img = plot_confusion_matrix_from_file(CONFUSION_MATRIX_FILE)
            st.image(img, use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")
    except Exception as e:
        st.error(f"Unable to load confusion matrix: {e}")

    st.header("Per-class Performance")
    if class_report is not None:
        try:
            st.dataframe(class_report)
            fig = plot_class_report(class_report)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Failed to display classification report: {e}")
    else:
        st.info("Classification report not available.")

    st.markdown(
        "---\n" "Interpretation: The model meets the business requirements if "
        "the validation accuracy and macro F1-score exceed target thresholds."
    )
