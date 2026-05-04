"""Chemical-Disease-Gene (CDG) dataset adapter for biomedical relation extraction."""

from __future__ import annotations

from medre_bench.datasets.base import BaseDataset, RelationExample
from medre_bench.datasets.preprocessing import NO_RELATION, process_bigbio_kb_doc
from medre_bench.registry import DATASET_REGISTRY

_LABEL_NAMES = [
    NO_RELATION,
    "chem_disease:marker/mechanism",
    "chem_disease:therapeutic",
    "chem_gene:activity:increases",
    "chem_gene:activity:decreases",
    "chem_gene:activity:affects",
    "chem_gene:binding:affects",
    "chem_gene:expression:increases",
    "chem_gene:expression:decreases",
    "chem_gene:expression:affects",
    "chem_gene:localization:affects",
    "chem_gene:metabolic_processing:increases",
    "chem_gene:metabolic_processing:decreases",
    "chem_gene:metabolic_processing:affects",
    "chem_gene:transport:increases",
    "chem_gene:transport:decreases",
    "chem_gene:transport:affects",
    "gene_disease:marker/mechanism",
    "gene_disease:therapeutic",
]

_LABEL_TO_ID = {label: idx for idx, label in enumerate(_LABEL_NAMES)}

_VAL_FRACTION = 0.15
_SEED = 42


@DATASET_REGISTRY.register("chem_dis_gene")
class ChemDisGeneDataset(BaseDataset):
    """ChemDisGene: Chemical-Disease-Gene relation extraction dataset."""

    HF_DATASET_ID = "bigbio/chem_dis_gene"
    HF_CONFIG = "chem_dis_gene_bigbio_kb"

    def name(self) -> str:
        return "chem_dis_gene"

    def num_labels(self) -> int:
        return len(_LABEL_NAMES)

    def label_names(self) -> list[str]:
        return list(_LABEL_NAMES)

    def load_split(self, split: str) -> list[RelationExample]:
        import random

        all_examples = self._load_raw_split("train")
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

    def _load_raw_split(self, split: str) -> list[RelationExample]:
        from datasets import load_dataset

        ds = load_dataset(self.HF_DATASET_ID, self.HF_CONFIG, split=split, trust_remote_code=True)

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
                f"No examples loaded from {self.HF_DATASET_ID}/{self.HF_CONFIG} split={split}."
            )

        return examples
