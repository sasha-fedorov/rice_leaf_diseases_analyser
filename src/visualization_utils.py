import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_training_curves(history: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["loss"], label="train_loss", marker="o")
    if "val_loss" in history.columns:
        axes[0].plot(history["val_loss"], label="val_loss", marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["accuracy"], label="train_accuracy", marker="o")
    if "val_accuracy" in history.columns:
        axes[1].plot(history["val_accuracy"], label="val_accuracy", marker="o")
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_confusion_matrix_from_file(path):
    from PIL import Image
    return Image.open(path)


def plot_class_report(report: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    report_reset = report.reset_index()
    index_col_name = report_reset.columns[0]
    sns.barplot(
        data=report_reset,
        x="f1-score",
        y=index_col_name,
        hue=index_col_name,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("F1 Score by Class")
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Class")
    fig.tight_layout()
    return fig


def plot_class_distribution(distribution: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    distribution = distribution.sort_values("Total Images", ascending=False)
    ax.barh(
        distribution["Class"],
        distribution["Total Images"],
        color="#4caf50"
    )
    ax.set_title("Dataset Class Distribution")
    ax.set_xlabel("Number of Images")
    ax.set_ylabel("Class")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig
