import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_training_curves(history: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["loss"], label="train_loss")
    if "val_loss" in history.columns:
        axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="train_accuracy")
    if "val_accuracy" in history.columns:
        axes[1].plot(history["val_accuracy"], label="val_accuracy")
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    return fig


def plot_confusion_matrix_from_file(path):
    from PIL import Image
    return Image.open(path)


def plot_class_report(report: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=report,
        x="f1-score",
        y=report.index,
        palette="viridis",
        ax=ax,
    )
    ax.set_title("F1 Score by Class")
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Class")
    return fig
