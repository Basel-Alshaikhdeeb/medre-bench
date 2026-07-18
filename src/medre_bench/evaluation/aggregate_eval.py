"""Evaluate an aggregate-trained checkpoint on each source dataset's own split.

The aggregate model has a 6-class head; each source dataset has its own label
space. To make the two comparable to the per-dataset baselines we collapse
both to binary (label 0 = NO_RELATION, else = RELATION) and report per-source
binary metrics.

Source examples whose entity types fall outside the CHEMICAL / DISEASE / GENE
schema are dropped from evaluation (they were never in the aggregate training
distribution). All other examples - including source-positive examples whose
type-pair falls outside the five aggregate classes, e.g. DISEASE-DISEASE - are
still scored: the ground truth is 'positive relation exists' regardless of the
five-class scheme, and any non-zero prediction from the aggregate model counts
as 'positive'.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from medre_bench.datasets.aggregate import _ENTITY_TYPE_MAP
from medre_bench.datasets.base import BaseDataset, RelationExample
from medre_bench.datasets.preprocessing import BINARY_LABEL_NAMES
from medre_bench.registry import DATASET_REGISTRY
from medre_bench.utils.logging import setup_logger

logger = setup_logger(__name__)


def _canonicalize_example(ex: RelationExample) -> RelationExample | None:
    """Return a copy of ``ex`` with entity types remapped to CHEMICAL/DISEASE/GENE.

    Preserves the original label unchanged - callers collapse to binary later.
    Returns None if either entity type is out of schema (drop from evaluation).
    """
    ct1 = _ENTITY_TYPE_MAP.get(ex.entity1_type)
    ct2 = _ENTITY_TYPE_MAP.get(ex.entity2_type)
    if ct1 is None or ct2 is None:
        return None
    return replace(ex, entity1_type=ct1, entity2_type=ct2)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Micro/macro F1 + precision/recall/accuracy at binary granularity."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
    )

    return {
        "n": int(len(y_true)),
        "n_positive_true": int(y_true.sum()),
        "n_positive_pred": int(y_pred.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_positive": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_positive": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def run_aggregate_evaluation(
    checkpoint_path: str,
    split: str = "test",
    sources: Optional[list[str]] = None,
    batch_size: int = 32,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Score an aggregate checkpoint on each source dataset's original split.

    Args:
        checkpoint_path: Path to the aggregate model checkpoint directory.
        split: Which split of each source to evaluate ('validation' or 'test').
        sources: Restrict to a subset of source datasets; defaults to all seven
            (all sources used to train the aggregate).
        batch_size: DataLoader batch size for inference.
        output_dir: If given, write per-source metrics JSON + a combined report.

    Returns:
        Nested dict {source_name: binary_metrics, "combined": binary_metrics}.
    """
    from transformers import AutoTokenizer, DataCollatorWithPadding

    from medre_bench.datasets.aggregate import AggregateDataset
    from medre_bench.datasets.base import apply_entity_markers
    from medre_bench.models.base import get_entity_marker_tokens
    from medre_bench.registry import MODEL_REGISTRY
    from medre_bench.training.trainer import REModel, RETokenizedDataset

    import medre_bench.datasets  # noqa: F401 - registration
    import medre_bench.models  # noqa: F401

    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")

    # Locate config_snapshot.yaml the same way the predictor / evaluator does.
    import yaml
    candidates = [
        ckpt / "config_snapshot.yaml",
        ckpt.parent / "config_snapshot.yaml",
        ckpt.parent.parent / "config_snapshot.yaml",
        ckpt.parent.parent.parent / "config_snapshot.yaml",
    ]
    config_path = next((p for p in candidates if p.exists()), None)
    if config_path is None:
        raise FileNotFoundError(
            f"Cannot find config_snapshot.yaml near {ckpt}; needed to recover "
            "model architecture and entity-marker strategy."
        )

    with open(config_path) as f:
        saved_config = yaml.safe_load(f)
    ckpt_dataset = saved_config.get("dataset", {}).get("name")
    if ckpt_dataset != "aggregate":
        logger.warning(
            f"config_snapshot.yaml reports dataset={ckpt_dataset!r}, expected "
            "'aggregate'. Proceeding, but per-source evaluation only makes sense "
            "for aggregate-trained checkpoints."
        )
    entity_marker_strategy = saved_config.get("model", {}).get(
        "entity_marker_strategy", "typed_entity_marker_punct"
    )
    max_seq_length = int(saved_config.get("model", {}).get("max_seq_length", 512))
    model_name = saved_config.get("model", {}).get("name", "bert-base")

    # Aggregate always has 6 output classes.
    agg = AggregateDataset()
    num_labels = agg.num_labels()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model_cls = MODEL_REGISTRY.get(model_name)
    base_model = model_cls()
    entity_marker_tokens = get_entity_marker_tokens(entity_marker_strategy)
    base_model.build(
        num_labels=num_labels,
        entity_marker_tokens=entity_marker_tokens if entity_marker_tokens else None,
    )
    base_model.tokenizer = tokenizer
    model = REModel(base_model, num_labels=num_labels)

    safetensors_path = ckpt / "model.safetensors"
    pytorch_path = ckpt / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(safetensors_path), device=str(device))
    elif pytorch_path.exists():
        state = torch.load(pytorch_path, map_location=device, weights_only=True)
    else:
        raise FileNotFoundError(f"No weights in {ckpt}")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    source_names = sources or list(AggregateDataset.SOURCE_DATASETS)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    per_source: dict[str, dict[str, Any]] = {}
    all_true: list[int] = []
    all_pred: list[int] = []

    for ds_name in source_names:
        try:
            source_cls = DATASET_REGISTRY.get(ds_name)
        except KeyError:
            logger.warning(f"Source {ds_name!r} not registered; skipping")
            continue
        source: BaseDataset = source_cls()
        # DrugProt has no labeled test split; the adapter aliases test -> validation.
        # Flag this to the user so the report row isn't silently mislabeled.
        effective_split = split
        if ds_name == "drugprot" and split == "test":
            logger.info(
                "drugprot has no labeled 'test' split; scoring on 'validation' instead "
                "(BigBio does not ship the shared-task test set)."
            )
            effective_split = "validation"
        try:
            raw = source.load_split(effective_split)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not load {ds_name}/{effective_split}: {exc}")
            continue

        y_true_bin: list[int] = []
        canon_examples: list[RelationExample] = []
        for ex in raw:
            can = _canonicalize_example(ex)
            if can is None:
                continue
            # Trainer's tokenizer applies markers itself when spans are valid;
            # keep the source label untouched, we collapse to binary after scoring.
            canon_examples.append(can)
            y_true_bin.append(0 if ex.label_id == 0 else 1)

        if not canon_examples:
            logger.warning(f"{ds_name}/{split}: no in-schema examples; skipping")
            continue

        eval_ds = RETokenizedDataset(
            canon_examples, tokenizer, max_seq_length, entity_marker_strategy
        )
        loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

        y_pred_bin: list[int] = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"eval {ds_name}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                argmax = torch.argmax(out.logits, dim=-1).cpu().numpy()
                y_pred_bin.extend((argmax != 0).astype(int).tolist())

        y_true_arr = np.array(y_true_bin)
        y_pred_arr = np.array(y_pred_bin)
        metrics = _binary_metrics(y_true_arr, y_pred_arr)
        metrics["n_original"] = int(len(raw))
        metrics["n_scored"] = int(len(canon_examples))
        metrics["n_dropped_out_of_schema"] = int(len(raw) - len(canon_examples))
        per_source[ds_name] = metrics
        all_true.extend(y_true_arr.tolist())
        all_pred.extend(y_pred_arr.tolist())

        logger.info(
            f"[aggregate-eval/{split}] {ds_name}: "
            f"n={metrics['n']} micro_f1={metrics['micro_f1']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"P+={metrics['precision_positive']:.4f} R+={metrics['recall_positive']:.4f}"
        )

    combined = _binary_metrics(np.array(all_true), np.array(all_pred)) if all_true else {}
    result = {"per_source": per_source, "combined": combined, "split": split, "checkpoint": str(ckpt)}

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"aggregate_eval_{split}.json").write_text(json.dumps(result, indent=2))
        logger.info(f"Wrote per-source metrics to {out / f'aggregate_eval_{split}.json'}")

    return result
