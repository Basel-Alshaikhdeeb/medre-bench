"""Sentence / text-level inference for a trained RE checkpoint.

Public entry point: :func:`run_prediction`. Given a sentence or a multi-sentence
text plus two entity strings, splits the text into sentences, locates both
entities in each, marks them with the checkpoint's entity-marker strategy,
and returns per-sentence + document-level relation predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from medre_bench.datasets.base import apply_entity_markers
from medre_bench.datasets.preprocessing import BINARY_LABEL_NAMES, split_into_sentences
from medre_bench.utils.logging import setup_logger

logger = setup_logger(__name__)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _find_entity_span(
    sentence: str, entity: str, forbidden: tuple[int, int] | None = None
) -> tuple[int, int] | None:
    """Case-insensitive substring search; skip matches overlapping ``forbidden``.

    Returns (start, end) or None if no non-overlapping occurrence exists.
    """
    if not entity:
        return None
    lower_sentence = sentence.lower()
    lower_entity = entity.lower()
    start = 0
    while True:
        idx = lower_sentence.find(lower_entity, start)
        if idx < 0:
            return None
        end = idx + len(entity)
        if forbidden is None or end <= forbidden[0] or idx >= forbidden[1]:
            return idx, end
        start = idx + 1


def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Reconstruct the trained REModel from a saved checkpoint directory.

    Mirrors :func:`medre_bench.evaluation.evaluator.run_evaluation`'s loading logic;
    reads model / dataset / binary_mode / entity-marker settings from
    ``config_snapshot.yaml`` (searched in the checkpoint dir and its parents).
    """
    import yaml
    from transformers import AutoTokenizer

    import medre_bench.datasets  # noqa: F401 - trigger dataset registration
    import medre_bench.models  # noqa: F401 - trigger model registration
    from medre_bench.models.base import get_entity_marker_tokens
    from medre_bench.registry import DATASET_REGISTRY, MODEL_REGISTRY
    from medre_bench.training.trainer import REModel

    config_candidates = [
        checkpoint_path / "config_snapshot.yaml",
        checkpoint_path.parent / "config_snapshot.yaml",
        checkpoint_path.parent.parent / "config_snapshot.yaml",
        checkpoint_path.parent.parent.parent / "config_snapshot.yaml",
    ]
    config_path = next((p for p in config_candidates if p.exists()), None)
    if config_path is None:
        raise FileNotFoundError(
            f"Cannot find config_snapshot.yaml near {checkpoint_path}. "
            "Expected in the checkpoint dir or one of its parents."
        )

    with open(config_path) as f:
        saved_config = yaml.safe_load(f)

    model_cfg = saved_config.get("model", {})
    dataset_cfg = saved_config.get("dataset", {})
    entity_marker_strategy = model_cfg.get("entity_marker_strategy", "typed_entity_marker_punct")
    max_seq_length = int(model_cfg.get("max_seq_length", 512))
    model_name = model_cfg.get("name", "bert-base")
    dataset_name = dataset_cfg.get("name")
    binary_mode = bool(dataset_cfg.get("binary_mode", False))

    if binary_mode:
        num_labels = 2
        label_names = list(BINARY_LABEL_NAMES)
    else:
        if not dataset_name:
            raise ValueError(
                "config_snapshot.yaml does not record dataset.name; "
                "cannot recover label set for a non-binary checkpoint."
            )
        dataset_cls = DATASET_REGISTRY.get(dataset_name)
        dataset = dataset_cls()
        num_labels = dataset.num_labels()
        label_names = dataset.label_names()

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

    model_cls = MODEL_REGISTRY.get(model_name)
    base_model = model_cls()
    entity_marker_tokens = get_entity_marker_tokens(entity_marker_strategy)
    base_model.build(
        num_labels=num_labels,
        entity_marker_tokens=entity_marker_tokens if entity_marker_tokens else None,
    )
    base_model.tokenizer = tokenizer
    model = REModel(base_model, num_labels=num_labels)

    safetensors_path = checkpoint_path / "model.safetensors"
    pytorch_path = checkpoint_path / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path), device=str(device))
    elif pytorch_path.exists():
        state = torch.load(pytorch_path, map_location=device, weights_only=True)
    else:
        raise FileNotFoundError(
            f"No model weights in {checkpoint_path}: expected model.safetensors "
            "or pytorch_model.bin."
        )
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, {
        "entity_marker_strategy": entity_marker_strategy,
        "max_seq_length": max_seq_length,
        "num_labels": num_labels,
        "label_names": label_names,
        "binary_mode": binary_mode,
        "dataset_name": dataset_name,
        "model_name": model_name,
    }


def _score_sentence(
    sentence: str,
    entity1: str,
    entity1_type: str,
    entity2: str,
    entity2_type: str,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    entity_marker_strategy: str,
    max_seq_length: int,
) -> tuple[np.ndarray, dict[str, list[int]]] | None:
    """Locate both entities in ``sentence`` and run one forward pass.

    Returns (softmax probs, spans) or None if either entity cannot be located
    without overlap.
    """
    e1_span = _find_entity_span(sentence, entity1)
    if e1_span is None:
        return None
    e2_span = _find_entity_span(sentence, entity2, forbidden=e1_span)
    if e2_span is None:
        return None

    marked = apply_entity_markers(
        text=sentence,
        e1_start=e1_span[0], e1_end=e1_span[1], e1_type=entity1_type,
        e2_start=e2_span[0], e2_end=e2_span[1], e2_type=entity2_type,
        strategy=entity_marker_strategy,
    )

    enc = tokenizer(
        marked,
        max_length=max_seq_length,
        padding=False,
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = out.logits[0].detach().cpu().float().numpy()
    probs = _softmax(logits, axis=-1)
    return probs, {"entity1_span": list(e1_span), "entity2_span": list(e2_span)}


def run_prediction(
    checkpoint_path: str,
    text: str,
    entity1: str,
    entity2: str,
    entity1_type: str = "ENTITY",
    entity2_type: str = "ENTITY",
    top_k: int = 5,
) -> dict[str, Any]:
    """Run inference on a sentence or multi-sentence text.

    For each sentence found via :func:`split_into_sentences`, both entity
    strings are located case-insensitively; sentences containing both are
    scored and their top-k label probabilities returned. A document-level
    binary decision (RELATION if any sentence votes RELATION) is also produced.
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")

    model, tokenizer, cfg = _load_checkpoint(ckpt, device)
    logger.info(
        f"Loaded {cfg['model_name']} (dataset={cfg['dataset_name']}, "
        f"binary_mode={cfg['binary_mode']}, {cfg['num_labels']} labels) on {device}"
    )

    sentence_spans = split_into_sentences(text)
    predictions: list[dict[str, Any]] = []

    for sent_start, sent_end in sentence_spans:
        sentence = text[sent_start:sent_end]
        scored = _score_sentence(
            sentence=sentence,
            entity1=entity1, entity1_type=entity1_type,
            entity2=entity2, entity2_type=entity2_type,
            tokenizer=tokenizer, model=model, device=device,
            entity_marker_strategy=cfg["entity_marker_strategy"],
            max_seq_length=cfg["max_seq_length"],
        )
        if scored is None:
            continue
        probs, spans = scored

        k = min(top_k, len(probs))
        top_indices = np.argsort(-probs)[:k]
        top_predictions = [
            {"label": cfg["label_names"][int(i)], "prob": float(probs[int(i)])}
            for i in top_indices
        ]
        argmax_id = int(np.argmax(probs))
        binary_decision = "NO_RELATION" if argmax_id == 0 else "RELATION"

        predictions.append({
            "sentence": sentence,
            "sentence_offset": [sent_start, sent_end],
            "entity1": {"text": entity1, "type": entity1_type, "span": spans["entity1_span"]},
            "entity2": {"text": entity2, "type": entity2_type, "span": spans["entity2_span"]},
            "binary": binary_decision,
            "p_relation": float(1.0 - probs[0]),
            "argmax_label": cfg["label_names"][argmax_id],
            "top_k": top_predictions,
        })

    doc_binary = "RELATION" if any(p["binary"] == "RELATION" for p in predictions) else "NO_RELATION"
    max_p = max((p["p_relation"] for p in predictions), default=0.0)

    return {
        "checkpoint": str(ckpt),
        "model_name": cfg["model_name"],
        "dataset_name": cfg["dataset_name"],
        "binary_mode_checkpoint": cfg["binary_mode"],
        "entity_marker_strategy": cfg["entity_marker_strategy"],
        "entity1": entity1,
        "entity2": entity2,
        "document_prediction": {
            "binary": doc_binary,
            "max_p_relation": max_p,
            "n_matched_sentences": len(predictions),
            "n_total_sentences": len(sentence_spans),
        },
        "sentence_predictions": predictions,
    }
