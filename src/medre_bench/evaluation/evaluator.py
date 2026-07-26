"""Evaluation pipeline for trained models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from medre_bench.datasets.base import BaseDataset, apply_entity_markers
from medre_bench.registry import DATASET_REGISTRY
from medre_bench.training.metrics import compute_per_class_metrics
from medre_bench.utils.logging import setup_logger

logger = setup_logger(__name__)


def run_evaluation(
    checkpoint_path: str,
    dataset_name: str,
    split: str = "test",
    output_dir: Optional[str] = None,
    overrides: Optional[dict] = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate a trained checkpoint on a dataset split.

    Args:
        checkpoint_path: Path to the saved model checkpoint.
        dataset_name: Registry key of the dataset.
        split: Dataset split to evaluate on.
        output_dir: Directory to save predictions (optional).
        overrides: Config overrides (unused, for CLI compatibility).
        batch_size: Evaluation batch size.

    Returns:
        Dict of evaluation metrics.
    """
    import medre_bench.datasets  # noqa: F401

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(checkpoint_path)

    # Load config snapshot to recover model architecture, marker strategy, etc.
    # config_snapshot.yaml lives in the timestamp dir - two levels above best/ -
    # not three. Search a few candidate locations to be robust to checkpoint
    # paths passed at different depths (matches the pattern used by
    # predictor.py, aggregate_eval.py, and annotation/model_pool.py).
    import yaml

    entity_marker_strategy = "typed_entity_marker_punct"
    max_seq_length = 512
    saved_config: dict = {}

    for candidate in (
        checkpoint / "config_snapshot.yaml",
        checkpoint.parent / "config_snapshot.yaml",
        checkpoint.parent.parent / "config_snapshot.yaml",
        checkpoint.parent.parent.parent / "config_snapshot.yaml",
    ):
        if candidate.exists():
            with open(candidate) as f:
                saved_config = yaml.safe_load(f) or {}
            entity_marker_strategy = saved_config.get("model", {}).get(
                "entity_marker_strategy", entity_marker_strategy
            )
            max_seq_length = saved_config.get("model", {}).get(
                "max_seq_length", max_seq_length
            )
            logger.info(f"loaded config snapshot from {candidate}")
            break
    else:
        logger.warning(
            f"no config_snapshot.yaml found near {checkpoint}; "
            "falling back to bert-base + typed_entity_marker_punct + max_seq_length=512"
        )

    # Load dataset
    dataset_cls = DATASET_REGISTRY.get(dataset_name)
    dataset: BaseDataset = dataset_cls()
    examples = dataset.load_split(split)
    logger.info(f"Loaded {len(examples)} examples from {dataset_name}/{split}")

    # Mirror the trainer's binary collapse so labels match the checkpoint's head
    binary_mode = bool(saved_config.get("dataset", {}).get("binary_mode", False))
    if binary_mode:
        from medre_bench.datasets.preprocessing import BINARY_LABEL_NAMES, collapse_to_binary

        examples = collapse_to_binary(examples)
        num_labels = 2
        label_names = list(BINARY_LABEL_NAMES)
        logger.info("binary_mode=True from config snapshot: collapsed eval labels to 2 classes")
    else:
        num_labels = dataset.num_labels()
        label_names = dataset.label_names()

    # Load tokenizer and model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))

    from medre_bench.training.trainer import REModel, RETokenizedDataset

    # Load the model weights (prefer safetensors format)
    safetensors_path = checkpoint / "model.safetensors"
    pytorch_path = checkpoint / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        model_state = load_file(str(safetensors_path), device=str(device))
    elif pytorch_path.exists():
        model_state = torch.load(pytorch_path, map_location=device, weights_only=True)
    else:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint}. "
            "Expected model.safetensors or pytorch_model.bin"
        )

    # Determine model architecture from saved config
    import medre_bench.models  # noqa: F401
    from medre_bench.registry import MODEL_REGISTRY
    from medre_bench.models.base import align_encoder_vocab_to_state_dict, get_entity_marker_tokens

    model_name = saved_config.get("model", {}).get("name", "bert-base")
    model_cls = MODEL_REGISTRY.get(model_name)
    base_model = model_cls()

    entity_marker_tokens = get_entity_marker_tokens(entity_marker_strategy)
    base_model.build(
        num_labels=num_labels,
        entity_marker_tokens=entity_marker_tokens if entity_marker_tokens else None,
    )
    base_model.tokenizer = tokenizer

    model = REModel(base_model, num_labels=num_labels)
    align_encoder_vocab_to_state_dict(model, model_state)
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()

    # Tokenize (variable-length; collator pads each batch to its longest sequence)
    from transformers import DataCollatorWithPadding

    eval_dataset = RETokenizedDataset(
        examples, tokenizer, max_seq_length, entity_marker_strategy
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator
    )

    # Run inference — keep the full logits so compute_metrics can compute ROC-AUC
    all_logits: list[np.ndarray] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.detach().cpu().float().numpy())
            all_labels.extend(labels.numpy().tolist())

    all_logits_arr = np.concatenate(all_logits, axis=0)
    all_labels_arr = np.array(all_labels)
    all_preds = np.argmax(all_logits_arr, axis=-1)

    # compute_metrics unpacks (predictions, labels); pass logits so it can
    # softmax internally and report ROC-AUC alongside F1.
    from medre_bench.training.metrics import compute_metrics as _compute

    metrics = _compute((all_logits_arr, all_labels_arr))

    # Per-class metrics
    per_class = compute_per_class_metrics(all_labels_arr, all_preds, label_names)
    metrics["per_class"] = per_class["per_class"]

    logger.info(f"Results on {dataset_name}/{split}:")
    logger.info(f"  Micro F1: {metrics['micro_f1']:.4f}")
    logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"\n{per_class['classification_report']}")

    # Save predictions if output dir specified
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        predictions_data = []
        for i, ex in enumerate(examples):
            predictions_data.append({
                "text": ex.text,
                "entity1": ex.entity1,
                "entity2": ex.entity2,
                "true_label": ex.label,
                "predicted_label": label_names[int(all_preds[i])],
                "correct": bool(all_preds[i] == all_labels_arr[i]),
            })

        with open(out_path / f"predictions_{dataset_name}_{split}.json", "w") as f:
            json.dump(predictions_data, f, indent=2)

        with open(out_path / f"metrics_{dataset_name}_{split}.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Predictions saved to {out_path}")

    return metrics
