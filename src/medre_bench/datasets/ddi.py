"""DDI Corpus dataset adapter for drug-drug interaction extraction."""

from __future__ import annotations

from medre_bench.datasets.base import BaseDataset, RelationExample
from medre_bench.datasets.preprocessing import NO_RELATION, process_bigbio_kb_doc
from medre_bench.registry import DATASET_REGISTRY

_LABEL_NAMES = [
    NO_RELATION,
    "ADVISE",
    "EFFECT",
    "INT",
    "MECHANISM",
]

_LABEL_TO_ID = {label: idx for idx, label in enumerate(_LABEL_NAMES)}

_SPLIT_MAP = {
    "train": "train",
    "test": "test",
}

_VAL_FRACTION = 0.15
_SEED = 42


@DATASET_REGISTRY.register("ddi")
class DDIDataset(BaseDataset):
    """DDI Corpus: SemEval 2013 Task 9 drug-drug interaction dataset."""

    HF_DATASET_ID = "bigbio/ddi_corpus"
    HF_CONFIG = "ddi_corpus_bigbio_kb"

    def name(self) -> str:
        return "ddi"

    def num_labels(self) -> int:
        return len(_LABEL_NAMES)

    def label_names(self) -> list[str]:
        return list(_LABEL_NAMES)

    def load_split(self, split: str) -> list[RelationExample]:
        import random

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
        for doc in ds:
            examples.extend(
                process_bigbio_kb_doc(
                    doc=doc,
                    label_to_id=_LABEL_TO_ID,
                    no_relation_label=NO_RELATION,
                )
            )

        if not examples:
            raise ValueError(
                f"No examples loaded from {self.HF_DATASET_ID}/{self.HF_CONFIG} split={hf_split}."
            )

        return examples
