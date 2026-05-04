"""EU-ADR dataset adapter for drug-disease-target relation extraction."""

from __future__ import annotations

from medre_bench.datasets.base import BaseDataset, RelationExample
from medre_bench.datasets.preprocessing import to_sentence_level_examples
from medre_bench.registry import DATASET_REGISTRY

_LABEL_NAMES = [
    "NA",  # Negative Association (also serves as no-relation label)
    "PA",  # Positive Association
    "SA",  # Speculative Association
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

        if split == "test":
            return all_examples[:test_size]
        if split in ("validation", "dev"):
            return all_examples[test_size : test_size + val_size]
        if split == "train":
            return all_examples[test_size + val_size :]

        raise ValueError(f"Unknown split: {split}")

    def _load_all(self) -> list[RelationExample]:
        from datasets import load_dataset

        ds = load_dataset("bigbio/euadr", name="euadr_source", split="train")

        examples: list[RelationExample] = []
        for row in ds:
            text = (row.get("title", "") + " " + row.get("abstract", "")).strip()
            doc_id = row.get("pmid", "")
            annotations = row.get("annotations", [])
            if not annotations:
                continue

            entities: list[dict] = []
            for ann in annotations:
                parts = ann.strip().split("\t")
                if len(parts) < 10 or parts[2] != "concept":
                    continue
                entities.append({
                    "id": parts[8],
                    "text": parts[3],
                    "type": parts[9],
                    "start": int(parts[4]),
                    "end": int(parts[5]),
                })

            relation_pairs: dict[tuple[str, str], str] = {}
            for ann in annotations:
                parts = ann.strip().split("\t")
                if len(parts) < 11 or parts[2] != "relation":
                    continue
                rel_type = parts[10].strip()
                if rel_type not in _LABEL_TO_ID:
                    continue
                relation_pairs[(parts[3], parts[4])] = rel_type

            examples.extend(
                to_sentence_level_examples(
                    text=text,
                    entities=entities,
                    relation_pairs=relation_pairs,
                    label_to_id=_LABEL_TO_ID,
                    no_relation_label="NA",
                    doc_id=str(doc_id),
                )
            )

        if not examples:
            raise ValueError("No examples loaded from bigbio/euadr (euadr_source).")

        return examples
