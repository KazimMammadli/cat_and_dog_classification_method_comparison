import os
import numpy as np
import matplotlib.pyplot as plt

def plot_history(history_dict, title: str = "Training History", save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    epochs = range(1, len(history_dict["accuracy"]) + 1)

    # Accuracy
    axes[0].plot(epochs, history_dict["accuracy"], color="#2196F3", label="Train Accuracy", linewidth=2)
    if "val_accuracy" in history_dict:
        axes[0].plot(epochs, history_dict["val_accuracy"], color="#4CAF50", linestyle="--", label="Val Accuracy", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs, history_dict["loss"], color="#2196F3", label="Train Loss", linewidth=2)
    if "val_loss" in history_dict:
        axes[1].plot(epochs, history_dict["val_loss"], color="#F44336", linestyle="--", label="Val Loss", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Crossentropy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_confusion_matrix(cm, class_names=None, title: str = "Confusion Matrix", save_path: str = None):
    if class_names is None:
        class_names = ["Cat", "Dog"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
