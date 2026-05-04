"""DrugProt dataset adapter for chemical-gene relation extraction."""

from __future__ import annotations

from medre_bench.datasets.base import BaseDataset, RelationExample
from medre_bench.datasets.preprocessing import NO_RELATION, process_bigbio_kb_doc
from medre_bench.registry import DATASET_REGISTRY

_LABEL_NAMES = [
    NO_RELATION,
    "ACTIVATOR",
    "AGONIST",
    "AGONIST-ACTIVATOR",
    "AGONIST-INHIBITOR",
    "ANTAGONIST",
    "DIRECT-REGULATOR",
    "INDIRECT-DOWNREGULATOR",
    "INDIRECT-UPREGULATOR",
    "INHIBITOR",
    "PART-OF",
    "PRODUCT-OF",
    "SUBSTRATE",
    "SUBSTRATE_PRODUCT-OF",
]

_LABEL_TO_ID = {label: idx for idx, label in enumerate(_LABEL_NAMES)}

_SPLIT_MAP = {
    "train": "train",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
}


@DATASET_REGISTRY.register("drugprot")
class DrugProtDataset(BaseDataset):
    """DrugProt: BioCreative VII chemical-gene interaction dataset."""

    HF_DATASET_ID = "bigbio/drugprot"
    HF_CONFIG = "drugprot_bigbio_kb"

    def name(self) -> str:
        return "drugprot"

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
                    no_relation_label=NO_RELATION,
                )
            )

        return examples
