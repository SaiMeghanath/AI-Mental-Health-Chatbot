# src/evaluator.py
# Offline evaluation harness for emotion and sentiment classifiers
# Run this script directly: python -m src.evaluator

import json
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from .config import (
    GOEMOTIONS_SUPPORTED_LABELS,
    EVAL_RESULTS_PATH,
)
from .emotion_classifier import predict_emotion
from .sentiment_classifier import predict_sentiment


def evaluate_emotion_classifier(max_samples: int = 500) -> dict:
    """
    Evaluate emotion classifier on GoEmotions (filtered, single-label).

    Args:
        max_samples: Maximum samples per class to evaluate (for speed).

    Returns:
        dict containing accuracy, precision, recall, f1, confusion_matrix.
    """
    from datasets import load_dataset

    print("Loading GoEmotions dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")

    # Filter: keep only single-label samples whose label is in our 6 supported classes
    label_names = dataset.features["labels"].feature.names
    supported_indices = {label_names.index(l) for l in GOEMOTIONS_SUPPORTED_LABELS if l in label_names}

    filtered = [
        row for row in dataset
        if len(row["labels"]) == 1 and row["labels"][0] in supported_indices
    ]

    # Balance classes and cap samples
    from collections import defaultdict
    per_class = defaultdict(list)
    for row in filtered:
        per_class[row["labels"][0]].append(row)

    samples = []
    for idx, rows in per_class.items():
        samples.extend(rows[:max_samples])

    print(f"Evaluating on {len(samples)} samples...")
    y_true, y_pred = [], []

    for i, row in enumerate(samples):
        text = row["text"]
        true_label = label_names[row["labels"][0]]
        pred = predict_emotion(text)["label"]

        y_true.append(true_label)
        y_pred.append(pred)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(samples)} done")

    results = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=GOEMOTIONS_SUPPORTED_LABELS).tolist(),
        "labels": GOEMOTIONS_SUPPORTED_LABELS,
        "classification_report": classification_report(
            y_true, y_pred, labels=GOEMOTIONS_SUPPORTED_LABELS, zero_division=0
        ),
        "n_samples": len(samples),
    }

    print("\nEmotion Classifier Results:")
    print(f"  Accuracy:        {results['accuracy']}")
    print(f"  Macro Precision: {results['macro_precision']}")
    print(f"  Macro Recall:    {results['macro_recall']}")
    print(f"  Macro F1:        {results['macro_f1']}")
    print(results["classification_report"])

    return results


def evaluate_sentiment_classifier(max_samples: int = 600) -> dict:
    """
    Evaluate sentiment classifier on SST-2 with 3-class remapping.

    Args:
        max_samples: Maximum samples to evaluate.

    Returns:
        dict containing accuracy, precision, recall, f1, confusion_matrix.
    """
    from datasets import load_dataset

    print("Loading SST-2 dataset...")
    dataset = load_dataset("stanfordnlp/sst2", split="validation")

    samples = list(dataset)[:max_samples]
    print(f"Evaluating on {len(samples)} samples...")

    y_true, y_pred = [], []

    for i, row in enumerate(samples):
        text = row["sentence"]
        # SST-2: label 0 = negative, 1 = positive → map to our 3-class scheme
        true_label = "positive" if row["label"] == 1 else "negative"
        pred = predict_sentiment(text)["label"]

        y_true.append(true_label)
        y_pred.append(pred)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(samples)} done")

    labels = ["negative", "neutral", "positive"]
    results = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels), 4),
        "macro_recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
        "n_samples": len(samples),
    }

    print("\nSentiment Classifier Results:")
    print(f"  Accuracy:        {results['accuracy']}")
    print(f"  Macro Precision: {results['macro_precision']}")
    print(f"  Macro Recall:    {results['macro_recall']}")
    print(f"  Macro F1:        {results['macro_f1']}")
    print(results["classification_report"])

    return results


def run_full_evaluation():
    """Run both evaluations and save results to disk."""
    os.makedirs("data", exist_ok=True)

    print("=" * 60)
    print("EMOTION CLASSIFIER EVALUATION")
    print("=" * 60)
    emotion_results = evaluate_emotion_classifier()

    print("\n" + "=" * 60)
    print("SENTIMENT CLASSIFIER EVALUATION")
    print("=" * 60)
    sentiment_results = evaluate_sentiment_classifier()

    combined = {
        "emotion_classifier": emotion_results,
        "sentiment_classifier": sentiment_results,
    }

    with open(EVAL_RESULTS_PATH, "w") as f:
        # Remove non-serialisable classification_report string before JSON dump
        serialisable = {
            "emotion_classifier": {k: v for k, v in emotion_results.items() if k != "classification_report"},
            "sentiment_classifier": {k: v for k, v in sentiment_results.items() if k != "classification_report"},
        }
        json.dump(serialisable, f, indent=2)

    print(f"\nResults saved to {EVAL_RESULTS_PATH}")
    return combined


if __name__ == "__main__":
    run_full_evaluation()
