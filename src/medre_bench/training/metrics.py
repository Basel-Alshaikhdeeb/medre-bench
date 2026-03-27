"""Metric computation for relation extraction evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute metrics compatible with HuggingFace Trainer's compute_metrics callback.

    Args:
        eval_pred: EvalPrediction with predictions and label_ids arrays.

    Returns:
        Dict of metric name -> value.
    """
    predictions, labels = eval_pred
    if predictions.ndim > 1:
        preds = np.argmax(predictions, axis=-1)
    else:
        preds = predictions

    return {
        "micro_f1": f1_score(labels, preds, average="micro"),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_precision": precision_score(labels, preds, average="micro"),
        "micro_recall": recall_score(labels, preds, average="micro"),
        "macro_precision": precision_score(labels, preds, average="macro"),
        "macro_recall": recall_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
    }


def compute_per_class_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """Compute per-class precision, recall, F1.

    Returns:
        Dict with 'per_class' (dict of class -> metrics) and 'classification_report' (str).
    """
    if predictions.ndim > 1:
        preds = np.argmax(predictions, axis=-1)
    else:
        preds = predictions

    report_dict = classification_report(
        labels, preds, target_names=label_names, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        labels, preds, target_names=label_names, zero_division=0
    )

    per_class = {}
    for label_name in label_names:
        if label_name in report_dict:
            per_class[label_name] = {
                "precision": report_dict[label_name]["precision"],
                "recall": report_dict[label_name]["recall"],
                "f1": report_dict[label_name]["f1-score"],
                "support": report_dict[label_name]["support"],
            }

    return {
        "per_class": per_class,
        "classification_report": report_str,
    }
