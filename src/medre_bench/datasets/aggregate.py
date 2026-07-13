"""Aggregate dataset combining 8 source RE datasets into 5 unified relation types.

Unifies heterogeneous per-dataset entity type strings into three canonical
categories (CHEMICAL, DISEASE, GENE) and remaps every positive relation to
one of five aggregate classes based on the (type1, type2) pair. Negatives
(NO_RELATION examples from each source) are subsampled to match the total
positive count so the resulting corpus is 50% positive / 50% negative.

Aggregate label set (id 0 is no-relation, per codebase convention):

    0 NO_RELATION
    1 Gene-Disease Association          - {GENE, DISEASE}
    2 Gene-Chemical Association         - {GENE, CHEMICAL}
    3 Gene-Gene Interaction             - {GENE, GENE}
    4 Chemical-Disease Association      - {CHEMICAL, DISEASE}
    5 Chemical-Chemical Interaction     - {CHEMICAL, CHEMICAL}

Any positive example whose entity types do not both map to CHEMICAL / DISEASE
/ GENE, or whose pair is not one of the five above (e.g. DISEASE-DISEASE),
is dropped.
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


# Every source-dataset entity type string, mapped to one of three canonical types.
# Case- and whitespace-sensitive - keep exact strings as they appear in the source.
_ENTITY_TYPE_MAP: dict[str, str] = {
    # DISEASE
    "DiseaseOrPhenotypicFeature": "DISEASE",   # BioRED
    "Diseases & Disorders": "DISEASE",         # EuADR
    "Disease": "DISEASE",                      # GAD, BC5CDR, chem_dis_gene
    "DISEASE": "DISEASE",                      # GAD (already canonical)
    # CHEMICAL
    "ChemicalEntity": "CHEMICAL",              # BioRED
    "Chemicals & Drugs": "CHEMICAL",           # EuADR
    "DRUG": "CHEMICAL",                        # DDI
    "GROUP": "CHEMICAL",                       # DDI
    "BRAND": "CHEMICAL",                       # DDI
    "DRUG_N": "CHEMICAL",                      # DDI
    "CHEMICAL": "CHEMICAL",                    # ChemProt, DrugProt
    "Chemical": "CHEMICAL",                    # BC5CDR, chem_dis_gene
    # GENE
    "GeneOrGeneProduct": "GENE",               # BioRED
    "SequenceVariant": "GENE",                 # BioRED (variants folded into GENE)
    "Genes & Molecular Sequences": "GENE",     # EuADR
    "SNP & Sequence variations": "GENE",       # EuADR (variants folded into GENE)
    "GENE-Y": "GENE",                          # ChemProt, DrugProt
    "GENE-N": "GENE",                          # ChemProt, DrugProt
    "Gene": "GENE",                            # GAD, chem_dis_gene
}

_LABEL_NAMES: list[str] = [
    "NO_RELATION",
    "Gene-Disease Association",
    "Gene-Chemical Association",
    "Gene-Gene Interaction",
    "Chemical-Disease Association",
    "Chemical-Chemical Interaction",
]

_LABEL_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(_LABEL_NAMES)}

# Unordered {canonical_type1, canonical_type2} -> aggregate class name.
_PAIR_TO_CLASS: dict[frozenset, str] = {
    frozenset(["GENE", "DISEASE"]): "Gene-Disease Association",
    frozenset(["GENE", "CHEMICAL"]): "Gene-Chemical Association",
    frozenset(["GENE"]): "Gene-Gene Interaction",
    frozenset(["CHEMICAL", "DISEASE"]): "Chemical-Disease Association",
    frozenset(["CHEMICAL"]): "Chemical-Chemical Interaction",
}

# GAD is intentionally excluded: its sentence-level schema carries no entity
# offsets, so its examples cannot receive the unified CHEMICAL/DISEASE/GENE
# entity markers the aggregate model is trained with. Including GAD would mix
# marked (7 sources) and unmarked (GAD) inputs and degrade the aggregate model.
_SOURCE_DATASETS: list[str] = [
    "bc5cdr",
    "biored",
    "chem_dis_gene",
    "chemprot",
    "ddi",
    "drugprot",
    "euadr",
]

_SEED = 42


def _canonicalize_pair(t1: str, t2: str) -> str | None:
    """Return the aggregate class name for a pair of canonical types, or None."""
    key = frozenset([t1]) if t1 == t2 else frozenset([t1, t2])
    return _PAIR_TO_CLASS.get(key)


def _remap_positive(ex: RelationExample) -> RelationExample | None:
    """Return a copy of ``ex`` with canonical entity types and aggregate label,
    or None if either entity type is out-of-schema or the pair is not one of
    the five aggregate classes.
    """
    ct1 = _ENTITY_TYPE_MAP.get(ex.entity1_type)
    ct2 = _ENTITY_TYPE_MAP.get(ex.entity2_type)
    if ct1 is None or ct2 is None:
        return None
    agg_class = _canonicalize_pair(ct1, ct2)
    if agg_class is None:
        return None
    return replace(
        ex,
        entity1_type=ct1,
        entity2_type=ct2,
        label=agg_class,
        label_id=_LABEL_TO_ID[agg_class],
    )


def _remap_negative(ex: RelationExample) -> RelationExample | None:
    """Return a NO_RELATION copy with canonical types; drop if entity types
    or the pair fall outside the schema (keeps the negative pool consistent
    with the marker vocabulary the model sees for positives).
    """
    ct1 = _ENTITY_TYPE_MAP.get(ex.entity1_type)
    ct2 = _ENTITY_TYPE_MAP.get(ex.entity2_type)
    if ct1 is None or ct2 is None:
        return None
    if _canonicalize_pair(ct1, ct2) is None:
        return None
    return replace(
        ex,
        entity1_type=ct1,
        entity2_type=ct2,
        label="NO_RELATION",
        label_id=0,
    )


@DATASET_REGISTRY.register("aggregate")
class AggregateDataset(BaseDataset):
    """Combined 8-dataset corpus with 5 unified relation classes + NO_RELATION.

    ``load_split(split)`` applies the same aggregation logic to whichever split
    is requested (train / validation / test). Positives are collected from all
    source datasets, negatives are subsampled to match the total positive count
    so the returned corpus is roughly 1:1 positive:negative.
    """

    SOURCE_DATASETS: ClassVar[list[str]] = list(_SOURCE_DATASETS)

    def name(self) -> str:
        return "aggregate"

    def num_labels(self) -> int:
        return len(_LABEL_NAMES)

    def label_names(self) -> list[str]:
        return list(_LABEL_NAMES)

    def load_split(self, split: str) -> list[RelationExample]:
        rng = random.Random(_SEED)
        # Per-source pools so negative sampling can be stratified.
        pos_by_source: dict[str, list[RelationExample]] = {}
        neg_by_source: dict[str, list[RelationExample]] = {}

        for ds_name in self.SOURCE_DATASETS:
            try:
                dataset_cls = DATASET_REGISTRY.get(ds_name)
            except KeyError:
                logger.warning(f"Source dataset {ds_name!r} not in registry; skipping")
                continue

            ds = dataset_cls()
            try:
                examples = ds.load_split(split)
            except Exception as exc:  # noqa: BLE001 - source datasets vary in split availability
                logger.warning(f"Could not load {ds_name}/{split}: {exc}")
                continue

            src_pos: list[RelationExample] = []
            src_neg: list[RelationExample] = []
            n_pos_in = n_neg_in = 0
            for ex in examples:
                if ex.label_id == 0:
                    n_neg_in += 1
                    remapped = _remap_negative(ex)
                    if remapped is not None:
                        src_neg.append(remapped)
                else:
                    n_pos_in += 1
                    remapped = _remap_positive(ex)
                    if remapped is not None:
                        src_pos.append(remapped)
            pos_by_source[ds_name] = src_pos
            neg_by_source[ds_name] = src_neg
            logger.info(
                f"[aggregate/{split}] {ds_name}: "
                f"pos {len(src_pos)}/{n_pos_in}, neg {len(src_neg)}/{n_neg_in}"
            )

        # Stratified negative sampling: each source contributes at most as many
        # negatives as it has positives (natural 1:1 per source, so no single
        # source dominates the negative pool). Shortfall from sources that ran
        # out of negatives is redistributed to sources with headroom, weighted
        # by that source's remaining negative capacity.
        neg_quotas: dict[str, int] = {
            s: min(len(pos_by_source[s]), len(neg_by_source[s]))
            for s in pos_by_source
        }
        total_positive = sum(len(v) for v in pos_by_source.values())
        target_total_neg = total_positive
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

        per_source_summary = {
            s: {
                "pos": len(pos_by_source[s]),
                "neg_available": len(neg_by_source[s]),
                "neg_sampled": neg_quotas[s],
            }
            for s in pos_by_source
        }
        class_dist = Counter(ex.label for ex in combined)
        logger.info(
            f"[aggregate/{split}] combined: {len(combined):,} examples "
            f"({len(positives):,} positive + {len(sampled_negatives):,} negative); "
            f"per-source: {per_source_summary}; "
            f"class distribution: {dict(class_dist)}"
        )
        return combined
