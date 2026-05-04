"""ChemProt dataset adapter for chemical-protein relation extraction."""

from __future__ import annotations

from medre_bench.datasets.base import BaseDataset, RelationExample
from medre_bench.datasets.preprocessing import process_bigbio_kb_doc
from medre_bench.registry import DATASET_REGISTRY

# ChemProt relation classes (BioCreative VI evaluation groups)
# "Not" serves as the no-relation label.
_LABEL_NAMES = [
    "Not",            # Negative / no relation
    "Upregulator",    # CPR:3
    "Downregulator",  # CPR:4
    "Agonist",        # CPR:5
    "Antagonist",     # CPR:6
    "Substrate",      # CPR:9
]

_LABEL_TO_ID = {label: idx for idx, label in enumerate(_LABEL_NAMES)}

_LABEL_REMAP = {
    "Regulator": "Upregulator",
    "Cofactor": "Substrate",
    "Modulator": "Agonist",
    "Undefined": "Not",
    "Part_of": "Not",
}

_SPLIT_MAP = {
    "train": "train",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
}


@DATASET_REGISTRY.register("chemprot")
class ChemProtDataset(BaseDataset):
    """ChemProt: BioCreative VI chemical-protein interaction dataset."""

    HF_DATASET_ID = "bigbio/chemprot"
    HF_CONFIG = "chemprot_bigbio_kb"

    def name(self) -> str:
        return "chemprot"

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
                    no_relation_label="Not",
                    label_remap=_LABEL_REMAP,
                )
            )

        if not examples:
            raise ValueError(
                f"No examples loaded from {self.HF_DATASET_ID}/{self.HF_CONFIG} split={hf_split}."
            )

        return examples
