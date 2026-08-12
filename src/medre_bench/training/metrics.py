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
    roc_auc_score,
)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute metrics compatible with HuggingFace Trainer's compute_metrics callback.

    Args:
        eval_pred: EvalPrediction with predictions (logits) and label_ids arrays.

    Returns:
        Dict of metric name -> value, including ROC-AUC (binary or one-vs-rest macro/weighted).
    """
    predictions, labels = eval_pred
    if predictions.ndim > 1:
        probs = _softmax(predictions, axis=-1)
        preds = np.argmax(predictions, axis=-1)
    else:
        probs = None
        preds = predictions

    metrics: dict[str, float] = {
        "micro_f1": f1_score(labels, preds, average="micro"),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "micro_precision": precision_score(labels, preds, average="micro", zero_division=0),
        "micro_recall": recall_score(labels, preds, average="micro", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }

    if probs is not None:
        n_classes = probs.shape[-1]
        present = np.unique(labels)
        try:
            if n_classes == 2:
                metrics["roc_auc"] = roc_auc_score(labels, probs[:, 1])
            elif len(present) >= 2:
                metrics["roc_auc_macro"] = roc_auc_score(
                    labels, probs, multi_class="ovr", average="macro", labels=np.arange(n_classes)
                )
                metrics["roc_auc_weighted"] = roc_auc_score(
                    labels, probs, multi_class="ovr", average="weighted", labels=np.arange(n_classes)
                )
        except ValueError:
            # Eval labels missing one or more classes; AUC undefined for those.
            pass

    return metrics


def make_per_source_compute_metrics(source_ids: list[str]):
    """Build a ``compute_metrics`` closure that slices predictions per source.

    Used by :func:`run_training` when the eval dataset carries per-example
    ``source_of_origin`` metadata (currently: the ``aggregate_binary`` dataset).
    Returns the standard aggregate metrics plus:

    * one ``{src}_micro_f1`` / ``{src}_macro_f1`` / ``{src}_f1_positive`` /
      ``{src}_precision_positive`` / ``{src}_recall_positive`` / ``{src}_n``
      block per source.
    * ``across_source_avg_f1`` — the mean of per-source ``f1_positive`` values;
      recommended as ``metric_for_best_model``.

    ``source_ids`` must be aligned index-for-index with the eval dataset the
    Trainer sees; the caller is responsible for keeping that alignment (which
    is the default because Trainer's eval dataloader uses ``shuffle=False``).
    """
    unique_sources = sorted(set(source_ids))
    ids = np.asarray(source_ids)

    def _compute(eval_pred) -> dict[str, float]:
        base = compute_metrics(eval_pred)

        predictions, labels = eval_pred
        if predictions.ndim > 1:
            preds = np.argmax(predictions, axis=-1)
        else:
            preds = predictions
        labels = np.asarray(labels)

        per_source_f1_pos: list[float] = []
        for src in unique_sources:
            mask = ids == src
            n = int(mask.sum())
            if n == 0:
                continue
            src_labels = labels[mask]
            src_preds = preds[mask]

            micro = float(f1_score(src_labels, src_preds, average="micro", zero_division=0))
            macro = float(f1_score(src_labels, src_preds, average="macro", zero_division=0))
            f1_pos = float(f1_score(src_labels, src_preds, pos_label=1, zero_division=0))
            prec_pos = float(precision_score(src_labels, src_preds, pos_label=1, zero_division=0))
            rec_pos = float(recall_score(src_labels, src_preds, pos_label=1, zero_division=0))

            base[f"{src}_micro_f1"] = micro
            base[f"{src}_macro_f1"] = macro
            base[f"{src}_f1_positive"] = f1_pos
            base[f"{src}_precision_positive"] = prec_pos
            base[f"{src}_recall_positive"] = rec_pos
            base[f"{src}_n"] = n
            per_source_f1_pos.append(f1_pos)

        base["across_source_avg_f1"] = (
            float(np.mean(per_source_f1_pos)) if per_source_f1_pos else 0.0
        )
        return base

    return _compute


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
