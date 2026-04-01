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

_SPLIT_MAP = {
    "train": "train",
}

_VAL_FRACTION = 0.15
_SEED = 42


@DATASET_REGISTRY.register("euadr")
class EuADRDataset(BaseDataset):
    """EU-ADR: European Adverse Drug Reactions corpus."""

    HF_DATASET_ID = "bigbio/euadr"
    HF_CONFIG = "euadr_bigbio_kb"

    def name(self) -> str:
        return "euadr"

    def num_labels(self) -> int:
        return len(_LABEL_NAMES)

    def label_names(self) -> list[str]:
        return list(_LABEL_NAMES)

    def load_split(self, split: str) -> list[RelationExample]:
        import random

        # EU-ADR only has a train split; carve validation and test from it
        all_examples = self._load_raw_split("train")
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

        return self._load_raw_split(split)

    def _load_raw_split(self, split: str) -> list[RelationExample]:
        from datasets import load_dataset

        hf_split = _SPLIT_MAP.get(split, split)
        ds = load_dataset(self.HF_DATASET_ID, self.HF_CONFIG, split=hf_split, trust_remote_code=True)

        examples = []
        seen_rel_types = set()
        for doc in ds:
            text = " ".join([p["text"][0] for p in doc["passages"]])
            entities_by_id = {}
            for entity in doc["entities"]:
                entities_by_id[entity["id"]] = {
                    "text": entity["text"][0],
                    "type": entity["type"],
                    "start": entity["offsets"][0][0],
                    "end": entity["offsets"][0][1],
                }

            for relation in doc["relations"]:
                rel_type = relation["type"]
                seen_rel_types.add(rel_type)
                if rel_type not in _LABEL_TO_ID:
                    continue

                arg1_id = relation["arg1_id"]
                arg2_id = relation["arg2_id"]

                if arg1_id not in entities_by_id or arg2_id not in entities_by_id:
                    continue

                e1 = entities_by_id[arg1_id]
                e2 = entities_by_id[arg2_id]

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
                        metadata={"doc_id": doc["id"]},
                    )
                )

        if not examples:
            raise ValueError(
                f"No examples loaded from {self.HF_DATASET_ID}/{self.HF_CONFIG} "
                f"split={hf_split}. Relation types found in data: {seen_rel_types}. "
                f"Expected types: {set(_LABEL_TO_ID.keys())}"
            )

        return examples
