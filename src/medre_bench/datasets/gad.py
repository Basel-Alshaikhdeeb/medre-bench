"""GAD dataset adapter for gene-disease association extraction."""

from __future__ import annotations

from medre_bench.datasets.base import BaseDataset, RelationExample
from medre_bench.registry import DATASET_REGISTRY

_LABEL_NAMES = [
    "0",  # No association
    "1",  # Association
]

_LABEL_TO_ID = {label: idx for idx, label in enumerate(_LABEL_NAMES)}

_SPLIT_MAP = {
    "train": "train",
    "test": "test",
}

_VAL_FRACTION = 0.15
_SEED = 42


@DATASET_REGISTRY.register("gad")
class GADDataset(BaseDataset):
    """GAD: Genetic Association Database - binary gene-disease RE dataset."""

    HF_DATASET_ID = "bigbio/gad"
    HF_CONFIG = "gad_fold0_bigbio_text"

    def name(self) -> str:
        return "gad"

    def num_labels(self) -> int:
        return len(_LABEL_NAMES)

    def label_names(self) -> list[str]:
        return list(_LABEL_NAMES)

    def load_split(self, split: str) -> list[RelationExample]:
        import random

        # GAD has no validation split; carve one from train
        if split in ("validation", "dev"):
            all_train = self._load_raw_split("train")
            rng = random.Random(_SEED)
            rng.shuffle(all_train)
            val_size = int(len(all_train) * _VAL_FRACTION)
            return all_train[:val_size]
        if split == "train":
            all_train = self._load_raw_split("train")
            rng = random.Random(_SEED)
            rng.shuffle(all_train)
            val_size = int(len(all_train) * _VAL_FRACTION)
            return all_train[val_size:]

        return self._load_raw_split(split)

    def _load_raw_split(self, split: str) -> list[RelationExample]:
        from datasets import load_dataset

        hf_split = _SPLIT_MAP.get(split, split)
        ds = load_dataset(self.HF_DATASET_ID, self.HF_CONFIG, split=hf_split, trust_remote_code=True)

        examples = []
        for row in ds:
            text = row["text"]
            label_str = str(row["labels"][0]) if row["labels"] else "0"

            if label_str not in _LABEL_TO_ID:
                continue

            # GAD is a text-level classification task (sentence-level RE)
            # Entities are not explicitly annotated with offsets in the bigbio_text schema
            examples.append(
                RelationExample(
                    text=text,
                    entity1="",
                    entity1_type="GENE",
                    entity1_start=0,
                    entity1_end=0,
                    entity2="",
                    entity2_type="DISEASE",
                    entity2_start=0,
                    entity2_end=0,
                    label=label_str,
                    label_id=_LABEL_TO_ID[label_str],
                    metadata={"id": row["id"]},
                )
            )

        return examples

    def entity_marker_strategy(self) -> str:
        # GAD uses sentence-level classification, no entity markers needed
        return "none"
