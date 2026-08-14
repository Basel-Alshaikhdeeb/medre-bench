"""Binary aggregate dataset — one class per pair (RELATION / NO_RELATION).

Follows the simplification we agreed on: since the "type name" of a relation is
fully determined by the (canonical_type1, canonical_type2) entity pair,
training a 5- or 9-way head that reproduces that lookup wastes capacity. The
model only needs to answer one question — "is there a relation between these
two marked entities?" — and downstream code can name the type from the entity
types at inference time via a rule table.

Layout:

* Canonical entity types: CHEMICAL, DISEASE, GENE, VARIANT.
* Valid pairs (unordered): {C,D}, {C,C}, {C,G}, {D,G}, {G,G}, {C,V}, {D,V},
  {G,V}, {V,V}. Anything else (e.g. DISEASE-DISEASE) is dropped.
* GAD excluded (no entity offsets → no markers → not comparable).

Splits:

* ``train``: soft-ceiling per source on positives (default 5,000), stratified
  1:1 negatives per source (identical logic to :class:`AggregateDataset` but
  with binary labels baked in from data-construction time).
* ``validation`` / ``test``: natural distribution union across the 7 sources
  (NO negative subsampling, NO capping). Each example carries
  ``metadata['source_of_origin']`` so downstream code can slice per-source
  metrics at evaluation time. DrugProt's ``test`` aliases to ``validation``
  (BigBio never released a labeled DrugProt test set).
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace
from typing import ClassVar

from medre_bench.datasets.base import BaseDataset, RelationExample
from medre_bench.registry import DATASET_REGISTRY
from medre_bench.utils.logging import setup_logger

logger = setup_logger(__name__)


# Canonical types: CHEMICAL, DISEASE, GENE, VARIANT (new — was folded into GENE
# in the 5-class aggregate; kept separate here so the model can attend to
# variant-specific patterns via the entity marker).
_ENTITY_TYPE_MAP: dict[str, str] = {
    # DISEASE
    "DiseaseOrPhenotypicFeature": "DISEASE",   # BioRED
    "Diseases & Disorders": "DISEASE",         # EuADR
    "Disease": "DISEASE",                      # BC5CDR, chem_dis_gene, GAD
    "DISEASE": "DISEASE",                      # canonical
    # CHEMICAL
    "ChemicalEntity": "CHEMICAL",              # BioRED
    "Chemicals & Drugs": "CHEMICAL",           # EuADR
    "DRUG": "CHEMICAL",                        # DDI
    "GROUP": "CHEMICAL",                       # DDI
    "BRAND": "CHEMICAL",                       # DDI
    "DRUG_N": "CHEMICAL",                      # DDI
    "CHEMICAL": "CHEMICAL",                    # ChemProt, DrugProt
    "Chemical": "CHEMICAL",                    # BC5CDR, chem_dis_gene
    # GENE (variants live in VARIANT below, not here)
    "GeneOrGeneProduct": "GENE",               # BioRED
    "Genes & Molecular Sequences": "GENE",     # EuADR
    "GENE-Y": "GENE",                          # ChemProt (raw)
    "GENE-N": "GENE",                          # ChemProt (raw)
    "GENE": "GENE",                            # DrugProt (BigBio normalizes GENE-Y/GENE-N -> GENE)
    "Gene": "GENE",                            # chem_dis_gene, GAD
    # VARIANT — new canonical type
    "SequenceVariant": "VARIANT",              # BioRED
    "SNP & Sequence variations": "VARIANT",    # EuADR
}


# Unordered pair schema. Any pair not in this set is dropped.
_VALID_PAIRS: set[frozenset] = {
    frozenset(["CHEMICAL", "DISEASE"]),
    frozenset(["CHEMICAL"]),                    # CHEMICAL-CHEMICAL
    frozenset(["GENE", "CHEMICAL"]),
    frozenset(["GENE", "DISEASE"]),
    frozenset(["GENE"]),                        # GENE-GENE
    frozenset(["VARIANT", "DISEASE"]),
    frozenset(["VARIANT", "CHEMICAL"]),
    frozenset(["VARIANT", "GENE"]),
    frozenset(["VARIANT"]),                     # VARIANT-VARIANT
}


_SOURCE_DATASETS: list[str] = [
    "bc5cdr",
    "biored",
    "chem_dis_gene",
    "chemprot",
    "ddi",
    "drugprot",
    "euadr",
]

_POSITIVE_SOFT_CEILING = 5000
_SEED = 42

BINARY_LABELS: list[str] = ["NO_RELATION", "RELATION"]


def _is_valid_pair(t1: str, t2: str) -> bool:
    key = frozenset([t1]) if t1 == t2 else frozenset([t1, t2])
    return key in _VALID_PAIRS


def _canonicalize_example(ex: RelationExample, source: str) -> RelationExample | None:
    """Remap entity types + collapse to binary label. Returns None if out-of-schema."""
    ct1 = _ENTITY_TYPE_MAP.get(ex.entity1_type)
    ct2 = _ENTITY_TYPE_MAP.get(ex.entity2_type)
    if ct1 is None or ct2 is None:
        return None
    if not _is_valid_pair(ct1, ct2):
        return None
    binary_id = 0 if ex.label_id == 0 else 1
    new_metadata = {**(ex.metadata or {}), "source_of_origin": source}
    return replace(
        ex,
        entity1_type=ct1,
        entity2_type=ct2,
        label=BINARY_LABELS[binary_id],
        label_id=binary_id,
        metadata=new_metadata,
    )


@DATASET_REGISTRY.register("aggregate_binary")
class AggregateBinaryDataset(BaseDataset):
    """Binary relation-detection corpus unioned across 7 source datasets."""

    SOURCE_DATASETS: ClassVar[list[str]] = list(_SOURCE_DATASETS)
    POSITIVE_SOFT_CEILING: ClassVar[int] = _POSITIVE_SOFT_CEILING

    def name(self) -> str:
        return "aggregate_binary"

    def num_labels(self) -> int:
        return 2

    def label_names(self) -> list[str]:
        return list(BINARY_LABELS)

    def load_split(self, split: str) -> list[RelationExample]:
        if split == "train":
            return self._load_train()
        if split in ("validation", "test"):
            return self._load_eval(split)
        raise ValueError(f"Unknown split for aggregate_binary: {split!r}")

    # ── train: soft ceiling + stratified 1:1 negatives ─────────────────────
    def _load_train(self) -> list[RelationExample]:
        rng = random.Random(_SEED)
        pos_by_source: dict[str, list[RelationExample]] = {}
        neg_by_source: dict[str, list[RelationExample]] = {}

        for src in self.SOURCE_DATASETS:
            try:
                ds_cls = DATASET_REGISTRY.get(src)
            except KeyError:
                logger.warning(f"[aggregate_binary/train] {src!r} not in registry; skipping")
                continue
            try:
                examples = ds_cls().load_split("train")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[aggregate_binary/train] could not load {src}/train: {exc}")
                continue

            src_pos: list[RelationExample] = []
            src_neg: list[RelationExample] = []
            for ex in examples:
                canon = _canonicalize_example(ex, source=src)
                if canon is None:
                    continue
                (src_pos if canon.label_id == 1 else src_neg).append(canon)

            # Soft ceiling on positives (only capping ones that exceed it).
            if len(src_pos) > self.POSITIVE_SOFT_CEILING:
                src_pos = rng.sample(src_pos, self.POSITIVE_SOFT_CEILING)
            pos_by_source[src] = src_pos
            neg_by_source[src] = src_neg
            logger.info(
                f"[aggregate_binary/train] {src}: "
                f"pos_kept={len(src_pos)}, neg_available={len(src_neg)}"
            )

        # Stratified 1:1 negative sampling with shortfall redistribution
        # (identical policy to AggregateDataset).
        neg_quotas: dict[str, int] = {
            s: min(len(pos_by_source[s]), len(neg_by_source[s]))
            for s in pos_by_source
        }
        target_total_neg = sum(len(v) for v in pos_by_source.values())
        shortfall = target_total_neg - sum(neg_quotas.values())
        if shortfall > 0:
            headroom = {
                s: len(neg_by_source[s]) - neg_quotas[s]
                for s in neg_quotas
                if len(neg_by_source[s]) - neg_quotas[s] > 0
            }
            total_headroom = sum(headroom.values())
            if total_headroom > 0:
                allocated = 0
                items = sorted(headroom.items())
                for i, (s, h) in enumerate(items):
                    is_last = i == len(items) - 1
                    take = (shortfall - allocated) if is_last else int(shortfall * h / total_headroom)
                    take = min(take, h)
                    neg_quotas[s] += take
                    allocated += take

        sampled_negatives: list[RelationExample] = []
        for s, quota in neg_quotas.items():
            pool = neg_by_source[s]
            if quota >= len(pool):
                sampled_negatives.extend(pool)
            else:
                sampled_negatives.extend(rng.sample(pool, quota))

        positives = [ex for exs in pos_by_source.values() for ex in exs]
        combined = positives + sampled_negatives
        rng.shuffle(combined)

        per_source = {
            s: {
                "pos": len(pos_by_source[s]),
                "neg_available": len(neg_by_source[s]),
                "neg_sampled": neg_quotas[s],
            }
            for s in pos_by_source
        }
        logger.info(
            f"[aggregate_binary/train] combined: {len(combined):,} examples "
            f"({len(positives):,} positive + {len(sampled_negatives):,} negative); "
            f"per-source: {per_source}"
        )
        return combined

    # ── validation/test: natural distribution, source-tagged ───────────────
    def _load_eval(self, split: str) -> list[RelationExample]:
        out: list[RelationExample] = []
        for src in self.SOURCE_DATASETS:
            # DrugProt has no labeled test split; alias 'test' -> 'validation'.
            effective_split = split
            if src == "drugprot" and split == "test":
                effective_split = "validation"
                logger.info(
                    "[aggregate_binary/test] drugprot has no labeled test split; "
                    "scoring on validation"
                )
            try:
                ds_cls = DATASET_REGISTRY.get(src)
            except KeyError:
                continue
            try:
                examples = ds_cls().load_split(effective_split)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[aggregate_binary/{split}] could not load {src}/{effective_split}: {exc}"
                )
                continue
            for ex in examples:
                canon = _canonicalize_example(ex, source=src)
                if canon is not None:
                    out.append(canon)

        counts = Counter(ex.metadata["source_of_origin"] for ex in out)
        label_dist = Counter(ex.label for ex in out)
        logger.info(
            f"[aggregate_binary/{split}] {len(out):,} examples at natural distribution; "
            f"per-source: {dict(counts)}; labels: {dict(label_dist)}"
        )
        return out
