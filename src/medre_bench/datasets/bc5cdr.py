"""BC5CDR dataset adapter for chemical-disease relation extraction."""

from __future__ import annotations

from itertools import product

from medre_bench.datasets.base import BaseDataset, RelationExample
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

            # Collect positive relation pairs
            positive_pairs = set()
            for relation in doc["relations"]:
                rel_type = relation["type"]
                seen_rel_types.add(rel_type)
                if rel_type != "CID":
                    continue

                arg1_id = relation["arg1_id"]
                arg2_id = relation["arg2_id"]

                if arg1_id not in entities_by_id or arg2_id not in entities_by_id:
                    continue

                positive_pairs.add((arg1_id, arg2_id))
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
                        label="CID",
                        label_id=_LABEL_TO_ID["CID"],
                        metadata={"doc_id": doc["id"]},
                    )
                )

            # Generate negative examples from chemical-disease pairs without a relation
            chemicals = {eid: e for eid, e in entities_by_id.items() if e["type"] == "Chemical"}
            diseases = {eid: e for eid, e in entities_by_id.items() if e["type"] == "Disease"}

            for chem_id, dis_id in product(chemicals, diseases):
                if (chem_id, dis_id) in positive_pairs:
                    continue

                e1 = chemicals[chem_id]
                e2 = diseases[dis_id]

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
                        label="No_Relation",
                        label_id=_LABEL_TO_ID["No_Relation"],
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
