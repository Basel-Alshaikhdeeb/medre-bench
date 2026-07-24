"""End-to-end orchestrator for two-tier relation annotation.

Flow (all logic in one place so the file is short and the sequence is easy to
follow):

    1. Parse input.json, extract passages + biomedical entities
    2. Split each passage into sentences, dedup mentions, enumerate pairs
    3. Load aggregate model, score every pair (tier-1)
    4. For each tier-2 candidate model, batch every pair routed to it and
       collect its softmax over that dataset's labels (tier-2)
    5. Combine per pair via the emission rules:
         - Type = derived from entity type pair (deterministic)
         - Category = highest-confidence non-NO_RELATION tier-2 label, or
           "unspecified" if none
         - Emit if tier-1 argmax != NO_RELATION OR any tier-2 predicts positive
    6. Dedup (A,B) vs (B,A) by category_confidence
    7. Append relation entries to each passage's ``relations[]`` and write file
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from medre_bench.annotation.bioc_io import (
    Entity,
    Passage,
    iter_passages,
    load_bioc,
    save_bioc,
)
from medre_bench.annotation.model_pool import LoadedModel, ModelPool
from medre_bench.annotation.pair_enumeration import Pair, enumerate_pairs
from medre_bench.datasets.base import RelationExample, apply_entity_markers
from medre_bench.training.trainer import RETokenizedDataset
from medre_bench.utils.logging import setup_logger

logger = setup_logger(__name__)


# Tier-2 routing: which per-dataset checkpoints to consult for a canonical
# type pair. Keyed by frozenset so lookups are order-independent.
_TIER2_ROUTING: dict[frozenset, list[str]] = {
    frozenset(["CHEMICAL", "DISEASE"]): ["bc5cdr", "chem_dis_gene", "biored", "euadr"],
    frozenset(["CHEMICAL"]):            ["ddi", "biored", "euadr"],
    frozenset(["GENE", "CHEMICAL"]):    ["chemprot", "drugprot", "biored", "euadr"],
    frozenset(["GENE", "DISEASE"]):     ["chem_dis_gene", "biored", "euadr"],
    frozenset(["GENE"]):                ["biored", "euadr"],
}


def _tier2_candidates_for(pair: Pair) -> list[str]:
    return _TIER2_ROUTING.get(pair.canonical_pair_key, [])


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=axis, keepdims=True)


def _pair_to_example(pair: Pair) -> RelationExample:
    """Convert a Pair into the RelationExample the tokenizer + collator expect.

    Entity offsets are sentence-relative because the trainer's
    ``apply_entity_markers`` operates on the sentence text, not the whole passage.
    """
    sent = pair.sentence
    return RelationExample(
        text=sent.text,
        entity1=pair.e1.text,
        entity1_type=pair.e1.canonical_type,
        entity1_start=pair.e1.passage_start - sent.passage_start,
        entity1_end=pair.e1.passage_end - sent.passage_start,
        entity2=pair.e2.text,
        entity2_type=pair.e2.canonical_type,
        entity2_start=pair.e2.passage_start - sent.passage_start,
        entity2_end=pair.e2.passage_end - sent.passage_start,
        label="",  # unused at inference
        label_id=0,
    )


def _score_pairs(
    pairs: list[Pair],
    loaded: LoadedModel,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    """Run one loaded model over a list of pairs. Returns an (N, num_labels)
    softmax matrix aligned with the input order."""
    from transformers import DataCollatorWithPadding

    if not pairs:
        return np.zeros((0, loaded.num_labels), dtype=np.float32)

    examples = [_pair_to_example(p) for p in pairs]
    ds = RETokenizedDataset(
        examples, loaded.tokenizer, loaded.max_seq_length, loaded.entity_marker_strategy
    )
    collator = DataCollatorWithPadding(tokenizer=loaded.tokenizer, padding="longest")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            iids = batch["input_ids"].to(loaded.device)
            att = batch["attention_mask"].to(loaded.device)
            out = loaded.model(input_ids=iids, attention_mask=att)
            logits = out.logits.detach().cpu().float().numpy()
            all_probs.append(_softmax(logits, axis=-1))
    return np.concatenate(all_probs, axis=0)


def _rel_id(pair: Pair) -> str:
    """Deterministic 8-char hash-based relation id."""
    h = hashlib.sha1()
    payload = (
        f"{pair.passage.document_id}|"
        f"{pair.passage.doc_offset}|"
        f"{pair.e1.ann_id}|{pair.e2.ann_id}"
    )
    h.update(payload.encode("utf-8"))
    return f"R_{h.hexdigest()[:8]}"


def _combine(
    pair: Pair,
    tier1_probs: np.ndarray,          # shape (num_agg_labels,)
    tier2_probs: dict[str, tuple[LoadedModel, np.ndarray]],
) -> dict[str, Any] | None:
    """Apply the emission rules. Returns an entry dict or None to drop."""
    tier1_argmax = int(np.argmax(tier1_probs))
    tier1_emit = tier1_argmax != 0  # index 0 is NO_RELATION by convention
    tier1_confidence = float(1.0 - tier1_probs[0])

    best = None  # {'category': str, 'confidence': float, 'source': str}
    for ds_key, (loaded, probs) in tier2_probs.items():
        argmax_id = int(np.argmax(probs))
        if argmax_id == 0:
            continue
        conf = float(probs[argmax_id])
        if best is None or conf > best["confidence"]:
            best = {
                "category": loaded.label_names[argmax_id],
                "confidence": conf,
                "source": ds_key,
            }

    if not tier1_emit and best is None:
        return None
    return {
        "type": pair.general_type,
        "category": best["category"] if best else "unspecified",
        "type_confidence": tier1_confidence,
        "category_confidence": best["confidence"] if best else 0.0,
    }


def run_annotation(
    input_path: str,
    output_path: str | None,
    best_models_path: str,
    batch_size: int = 32,
    device: str = "auto",
    provider: str = "LCSB",
    single_checkpoint_at_a_time: bool = False,
) -> str:
    """Annotate the BioC file with relation predictions and write the result.

    Returns the resolved output path.
    """
    input_p = Path(input_path).expanduser().resolve()
    if not input_p.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_p}")

    if output_path:
        output_p = Path(output_path).expanduser().resolve()
    else:
        output_p = input_p.with_name(f"{input_p.stem}_annotated{input_p.suffix}")

    best_models_p = Path(best_models_path).expanduser().resolve()
    best_models_map = yaml.safe_load(best_models_p.read_text()) or {}
    if not isinstance(best_models_map, dict) or "aggregate" not in best_models_map:
        raise ValueError(
            f"{best_models_p}: expected a mapping with at least an 'aggregate' key"
        )

    # 1. Parse input
    raw = load_bioc(input_p)
    passages = iter_passages(raw)
    total_entities = sum(len(p.entities) for p in passages)
    logger.info(
        f"loaded {len(passages)} passages, {total_entities} biomedical entities "
        f"(after dropping non-biomedical annotations)"
    )

    # 2. Enumerate pairs
    all_pairs: list[Pair] = []
    for p in passages:
        all_pairs.extend(enumerate_pairs(p))
    logger.info(f"enumerated {len(all_pairs)} candidate pairs across sentences")
    if not all_pairs:
        # Nothing to score; still write out the file so downstream tools see it.
        save_bioc(raw, output_p)
        logger.info(f"no candidate pairs; wrote input verbatim to {output_p}")
        return str(output_p)

    # 3. Model pool
    pool = ModelPool(
        paths=best_models_map,
        device_preference=device,
        single_at_a_time=single_checkpoint_at_a_time,
    )

    # 4. Tier-1: score every pair with the aggregate model
    aggregate = pool.get("aggregate")
    tier1_probs = _score_pairs(all_pairs, aggregate, batch_size, desc="tier1:aggregate")

    # 5. Tier-2: for each dataset in the routing table, score every pair that routes to it
    tier2_probs: dict[int, dict[str, tuple[LoadedModel, np.ndarray]]] = defaultdict(dict)
    for ds_key in ["bc5cdr", "chem_dis_gene", "biored", "euadr", "ddi", "chemprot", "drugprot"]:
        if ds_key not in best_models_map:
            logger.warning(
                f"skipping tier-2 model {ds_key!r}: no path in best_models.yaml"
            )
            continue
        targets = [
            (idx, pair)
            for idx, pair in enumerate(all_pairs)
            if ds_key in _tier2_candidates_for(pair)
        ]
        if not targets:
            continue
        loaded = pool.get(ds_key)
        pairs_only = [pair for _, pair in targets]
        probs = _score_pairs(
            pairs_only, loaded, batch_size, desc=f"tier2:{ds_key}"
        )
        for (idx, _), row in zip(targets, probs):
            tier2_probs[idx][ds_key] = (loaded, row)

    # 6. Combine + emit; then dedup unordered pairs
    dedup: dict[tuple[str, int, tuple[str, str]], tuple[dict[str, Any], Pair]] = {}
    for i, pair in enumerate(all_pairs):
        entry = _combine(pair, tier1_probs[i], tier2_probs.get(i, {}))
        if entry is None:
            continue
        # Unordered dedup key: (doc, passage_offset, sorted-entity-ids)
        u_key = (
            pair.passage.document_id,
            pair.passage.doc_offset,
            tuple(sorted([pair.e1.ann_id, pair.e2.ann_id])),
        )
        existing = dedup.get(u_key)
        if existing is None or entry["category_confidence"] > existing[0]["category_confidence"]:
            dedup[u_key] = (entry, pair)

    logger.info(
        f"emitting {len(dedup)} relation entries "
        f"(from {len(all_pairs)} ordered candidates, deduped unordered)"
    )

    # 7. Attach to each passage's relations[] in place
    for u_key, (entry, pair) in dedup.items():
        rel = {
            "id": _rel_id(pair),
            "entity1_id": pair.e1.ann_id,
            "entity2_id": pair.e2.ann_id,
            "type": entry["type"],
            "category": entry["category"],
            "provider": provider,
            "type_confidence": round(entry["type_confidence"], 4),
            "category_confidence": round(entry["category_confidence"], 4),
        }
        pair.passage.raw.setdefault("relations", []).append(rel)

    save_bioc(raw, output_p)
    logger.info(f"wrote annotated BioC to {output_p}")
    return str(output_p)
