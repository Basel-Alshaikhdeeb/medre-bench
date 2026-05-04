"""BC5CDR dataset adapter for chemical-disease relation extraction."""

from __future__ import annotations

from medre_bench.datasets.base import BaseDataset, RelationExample
from medre_bench.datasets.preprocessing import process_bigbio_kb_doc
from medre_bench.registry import DATASET_REGISTRY

_LABEL_NAMES = [
    "No_Relation",
    "CID",  # Chemical-Induced Disease
]

_LABEL_TO_ID = {label: idx for idx, label in enumerate(_LABEL_NAMES)}

_SPLIT_MAP = {
    "train": "train",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
}


@DATASET_REGISTRY.register("bc5cdr")
class BC5CDRDataset(BaseDataset):
    """BC5CDR: BioCreative V Chemical-Disease Relation corpus."""

    HF_DATASET_ID = "bigbio/bc5cdr"
    HF_CONFIG = "bc5cdr_bigbio_kb"

    def name(self) -> str:
        return "bc5cdr"

    def num_labels(self) -> int:
        return len(_LABEL_NAMES)

    def label_names(self) -> list[str]:
        return list(_LABEL_NAMES)

    def load_split(self, split: str) -> list[RelationExample]:
        from datasets import load_dataset

        hf_split = _SPLIT_MAP.get(split, split)
        ds = load_dataset(self.HF_DATASET_ID, self.HF_CONFIG, split=hf_split, trust_remote_code=True)

        examples = []
        for doc in ds:
            examples.extend(
                process_bigbio_kb_doc(
                    doc=doc,
                    label_to_id=_LABEL_TO_ID,
                    no_relation_label="No_Relation",
                )
            )

        if not examples:
            raise ValueError(
                f"No examples loaded from {self.HF_DATASET_ID}/{self.HF_CONFIG} split={hf_split}."
            )

        return examples
