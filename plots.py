"""Vẽ learning curves + confusion matrix + phân tích error cho lab CIFAR-10."""
from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def plot_learning_curves(model_type: str) -> None:
    """Vẽ loss + accuracy curves từ file history JSON."""
    hist_path = os.path.join(RESULTS_DIR, f"{model_type}_history.json")
    if not os.path.exists(hist_path):
        print(f"[!] Không tìm thấy {hist_path}, bỏ qua learning curve.")
        return

    with open(hist_path, encoding="utf-8") as f:
        h = json.load(f)

    epochs = range(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, h["loss"], "o-", label="train", linewidth=2)
    if "val_loss" in h:
        axes[0].plot(epochs, h["val_loss"], "s-", label="validation", linewidth=2)
    axes[0].set_title(f"{model_type} — Loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, h["accuracy"], "o-", label="train", linewidth=2)
    if "val_accuracy" in h:
        axes[1].plot(epochs, h["val_accuracy"], "s-", label="validation", linewidth=2)
    axes[1].set_title(f"{model_type} — Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{model_type}_curves.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out}")


def plot_confusion_and_analyse(model_type: str) -> None:
    """Vẽ confusion matrix heatmap và in top-3 cặp class nhầm nhất."""
    model_path = os.path.join(MODEL_DIR, f"{model_type}_model.keras")
    data_path = os.path.join(DATA_DIR, "cifar10.npz")

    if not os.path.exists(model_path):
        print(f"[!] Không tìm thấy {model_path}, bỏ qua confusion matrix.")
        return

    data = np.load(data_path)
    x_test, y_test = data["x_test"], data["y_test"]

    model = tf.keras.models.load_model(model_path)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CIFAR_CLASSES, yticklabels=CIFAR_CLASSES,
        cbar_kws={"label": "Count"},
    )
    plt.title(f"{model_type} — Confusion Matrix (test set)")
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{model_type}_confusion.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved {out}")

    # Top-3 cặp class nhầm nhất (đối xứng: count[i→j] + count[j→i])
    cm_off = cm.copy().astype(int)
    np.fill_diagonal(cm_off, 0)
    pair_counts = []
    for i in range(10):
        for j in range(i + 1, 10):
            total = cm_off[i, j] + cm_off[j, i]
            pair_counts.append((total, i, j, cm_off[i, j], cm_off[j, i]))
    pair_counts.sort(reverse=True)

    print(f"\n  Top-3 cặp class nhầm nhất của {model_type}:")
    print(f"  {'Pair':<26} {'Total':>6} {'i→j':>5} {'j→i':>5}")
    for total, i, j, ij, ji in pair_counts[:3]:
        pair = f"{CIFAR_CLASSES[i]} ↔ {CIFAR_CLASSES[j]}"
        print(f"  {pair:<26} {total:>6} {ij:>5} {ji:>5}")


def main() -> None:
    if len(sys.argv) < 2:
        models = ["cifar_mlp", "cnn", "cnn_aug"]
    else:
        models = sys.argv[1:]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for m in models:
        print(f"\n=== {m} ===")
        plot_learning_curves(m)
        plot_confusion_and_analyse(m)


if __name__ == "__main__":
    main()