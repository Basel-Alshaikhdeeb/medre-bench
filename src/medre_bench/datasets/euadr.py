"""EU-ADR dataset adapter for drug-disease-target relation extraction."""

from __future__ import annotations

from medre_bench.datasets.base import BaseDataset, RelationExample
from medre_bench.registry import DATASET_REGISTRY

_LABEL_NAMES = [
    "PA",  # Positive Association
    "SA",  # Speculative Association
    "NA",  # Negative Association
]

_LABEL_TO_ID = {label: idx for idx, label in enumerate(_LABEL_NAMES)}

_VAL_FRACTION = 0.15
_SEED = 42


@DATASET_REGISTRY.register("euadr")
class EuADRDataset(BaseDataset):
    """EU-ADR: European Adverse Drug Reactions corpus."""

    def name(self) -> str:
        return "euadr"

    def num_labels(self) -> int:
        return len(_LABEL_NAMES)

    def label_names(self) -> list[str]:
        return list(_LABEL_NAMES)

    def load_split(self, split: str) -> list[RelationExample]:
        import random

        all_examples = self._load_all()
        rng = random.Random(_SEED)
        rng.shuffle(all_examples)

        val_size = int(len(all_examples) * _VAL_FRACTION)
        test_size = int(len(all_examples) * _VAL_FRACTION)

        if split in ("test",):
            return all_examples[:test_size]
        if split in ("validation", "dev"):
            return all_examples[test_size : test_size + val_size]
        if split == "train":
            return all_examples[test_size + val_size :]

        raise ValueError(f"Unknown split: {split}")

    def _load_all(self) -> list[RelationExample]:
        from datasets import load_dataset

        ds = load_dataset("bigbio/euadr", name="euadr_source", split="train")

        examples = []
        seen_rel_types = set()
        for row in ds:
            text = (row.get("title", "") + " " + row.get("abstract", "")).strip()
            doc_id = row.get("pmid", "")
            annotations = row.get("annotations", [])

            if not annotations:
                continue

            # First pass: collect concept (entity) annotations indexed by their ID
            concepts = {}
            for ann in annotations:
                parts = ann.strip().split("\t")
                if len(parts) < 3 or parts[2] != "concept":
                    continue
                # Format: rel_category, True/False, "concept", text, start, end, annotators, db_ids, idx, entity_type
                if len(parts) < 10:
                    continue
                idx = parts[8]
                concepts[idx] = {
                    "text": parts[3],
                    "type": parts[9],
                    "start": int(parts[4]),
                    "end": int(parts[5]),
                }

            # Second pass: collect relation annotations
            for ann in annotations:
                parts = ann.strip().split("\t")
                if len(parts) < 3 or parts[2] != "relation":
                    continue
                # Format: rel_category, True/False, "relation", ent1_idx, ent2_idx,
                #         db_ids1, db_ids2, offsets1, offsets2, annotators, label
                if len(parts) < 11:
                    continue

                rel_type = parts[10].strip()
                seen_rel_types.add(rel_type)
                if rel_type not in _LABEL_TO_ID:
                    continue

                ent1_idx = parts[3]
                ent2_idx = parts[4]

                if ent1_idx not in concepts or ent2_idx not in concepts:
                    continue

                e1 = concepts[ent1_idx]
                e2 = concepts[ent2_idx]

                examples.append(
                    RelationExample(
                        text=text,
                        entity1=e1["text"],
                        entity1_type=e1["type"],
                        entity1_start=e1["start"],
                        entity1_end=e1["end"],
                        entity2=e2["text"],
                        entity2_type=e2["type"],
                        entity2_start=e2["start"],
                        entity2_end=e2["end"],
                        label=rel_type,
                        label_id=_LABEL_TO_ID[rel_type],
                        metadata={"doc_id": doc_id},
                    )
                )

        if not examples:
            raise ValueError(
                f"No examples loaded from bigbio/euadr (euadr_source). "
                f"Relation types found in data: {seen_rel_types}. "
                f"Expected types: {set(_LABEL_TO_ID.keys())}"
            )

        return examples
