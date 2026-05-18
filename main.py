from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, Type, TypedDict

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from classifier import (
    LogisticRegressionClassifier,
    NeuralNetworkClassifier,
    MNISTClassifier,
    CIFARMLPClassifier,
    CIFARCNNClassifier,
    CIFARCNNAugClassifier,
    CIFARCNNImprovedClassifier,
)

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"


class ModelSpec(TypedDict):
    classifier: Type[MNISTClassifier]
    dataset: str


MODEL_TYPES: Dict[str, ModelSpec] = {
    "logistic": {"classifier": LogisticRegressionClassifier, "dataset": "mnist"},
    "nn": {"classifier": NeuralNetworkClassifier, "dataset": "mnist"},
    "cifar_mlp": {"classifier": CIFARMLPClassifier, "dataset": "cifar10"},
    "cnn": {"classifier": CIFARCNNClassifier, "dataset": "cifar10"},
    "cnn_aug": {"classifier": CIFARCNNAugClassifier, "dataset": "cifar10"},
    "cnn_improved": {"classifier": CIFARCNNImprovedClassifier, "dataset": "cifar10"},
}


def _model_path(model_type: str) -> str:
    return os.path.join(MODEL_DIR, f"{model_type}_model.keras")


def _data_path(dataset: str) -> str:
    return os.path.join(DATA_DIR, f"{dataset}.npz")


def _configure_mnist() -> None:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Flatten 28x28 images to 784-dim vectors and normalise to [0, 1]
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    np.savez(
        _data_path("mnist"),
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
    )
    print(f"Dataset saved to {_data_path('mnist')}")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples:     {len(x_test)}")


def _configure_cifar10() -> None:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Keep image tensors for CNN and normalise pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    np.savez(
        _data_path("cifar10"),
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
    )
    print(f"Dataset saved to {_data_path('cifar10')}")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples:     {len(x_test)}")


# ── configure ────────────────────────────────────────────────────────────────

def configure(args: argparse.Namespace) -> None:
    """Download and pre-process datasets used by the selected models."""

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.dataset in {"mnist", "all"}:
        _configure_mnist()
    if args.dataset in {"cifar10", "all"}:
        _configure_cifar10()


# ── train ────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """Train the selected model and save it to disk."""
    model_type: str = args.model
    spec = MODEL_TYPES[model_type]
    data_path = _data_path(spec["dataset"])
    if not os.path.exists(data_path):
        sys.exit(
            f"Error: Dataset not found at {data_path}. "
            f"Run 'configure --dataset {spec['dataset']}' first."
        )

    data = np.load(data_path)
    x_train, y_train = data["x_train"], data["y_train"]

    cls = spec["classifier"]
    classifier = cls()
    classifier.model = classifier.build_model()

    print(f"Training {model_type} model on {spec['dataset']} …")
    history = classifier.train(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = _model_path(model_type)
    classifier.save(save_path)
    print(f"Model saved to {save_path}")

    # Lưu history để vẽ learning curve
    os.makedirs(RESULTS_DIR, exist_ok=True)
    hist_path = os.path.join(RESULTS_DIR, f"{model_type}_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    print(f"History saved to {hist_path}")


# ── test ─────────────────────────────────────────────────────────────────────

def test(args: argparse.Namespace) -> None:
    """Evaluate the selected model and write results to a Markdown file."""
    model_type: str = args.model
    spec = MODEL_TYPES[model_type]
    data_path = _data_path(spec["dataset"])
    if not os.path.exists(data_path):
        sys.exit(
            f"Error: Dataset not found at {data_path}. "
            f"Run 'configure --dataset {spec['dataset']}' first."
        )

    model_path = _model_path(model_type)
    if not os.path.exists(model_path):
        sys.exit(f"Error: Trained model not found at {model_path}. Run 'train --model {model_type}' first.")

    data = np.load(data_path)
    x_test, y_test = data["x_test"], data["y_test"]

    cls = spec["classifier"]
    classifier = cls()
    classifier.load(model_path)

    print(f"Evaluating {model_type} model on {spec['dataset']} …")
    results = classifier.evaluate(x_test, y_test)

    y_pred = results["y_pred"]
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    num_weights = classifier.model.count_params()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Markdown report
    md_path = os.path.join(RESULTS_DIR, f"{model_type}_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {model_type.upper()} Model — Test Results\n\n")
        f.write(f"- **Dataset:** {spec['dataset']}\n")
        f.write(f"- **Loss:** {results['loss']:.4f}\n")
        f.write(f"- **Accuracy:** {acc:.4f}\n")
        f.write(f"- **Precision:** {precision:.4f}\n")
        f.write(f"- **Recall:** {recall:.4f}\n")
        f.write(f"- **F1-score:** {f1:.4f}\n")
        f.write(f"- **Weights:** {num_weights}\n\n")
        f.write("## Classification Report\n\n")
        f.write("```\n")
        f.write(report)
        f.write("```\n\n")
        f.write("## Confusion Matrix\n\n")
        f.write("```\n")
        f.write(np.array2string(cm, separator=", "))
        f.write("\n```\n")

    # JSON report
    json_path = os.path.join(RESULTS_DIR, f"{model_type}_results.json")
    json_data = {
        "model": model_type,
        "dataset": spec["dataset"],
        "weights": num_weights,
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Results saved to {md_path}")
    print(f"Results saved to {json_path}")


# ── summary ───────────────────────────────────────────────────────────────────

def summary(args: argparse.Namespace) -> None:
    """Read all JSON result files and write a combined summary.md."""
    json_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_results.json")))
    if not json_files:
        sys.exit("Error: No result JSON files found. Run 'test' first.")

    records = []
    for path in json_files:
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))

    md_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Image Classification — Summary\n\n")
        f.write("| Model | Dataset | Weights | Accuracy | Precision | Recall | F1-score |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in records:
            f.write(
                f"| {r['model']} | {r.get('dataset', 'mnist')} | {r['weights']:,} "
                f"| {r['accuracy']:.4f} | {r['precision']:.4f} "
                f"| {r['recall']:.4f} | {r['f1_score']:.4f} |\n"
            )

    print(f"Summary saved to {md_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Image Classification CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # configure
    configure_parser = subparsers.add_parser("configure", help="Download and prepare datasets")
    configure_parser.add_argument(
        "--dataset", default="mnist", choices=["mnist", "cifar10", "all"],
        help="Dataset to prepare (default: mnist)",
    )

    # train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model", required=True, choices=MODEL_TYPES.keys(),
        help="Model type to train (logistic | nn | cifar_mlp | cnn | cnn_aug | cnn_improved)",
    )
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")

    # test
    test_parser = subparsers.add_parser("test", help="Evaluate a trained model")
    test_parser.add_argument(
        "--model", required=True, choices=MODEL_TYPES.keys(),
        help="Model type to evaluate (logistic | nn | cifar_mlp | cnn | cnn_aug | cnn_improved)",
    )

    # summary
    subparsers.add_parser("summary", help="Generate summary.md from all test results")

    args = parser.parse_args()

    commands = {
        "configure": configure,
        "train": train,
        "test": test,
        "summary": summary,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
