"""ChemProt dataset adapter for chemical-protein relation extraction."""

from __future__ import annotations

from medre_bench.datasets.base import BaseDataset, RelationExample, apply_entity_markers
from medre_bench.registry import DATASET_REGISTRY

# ChemProt relation classes (BioCreative VI evaluation groups)
_LABEL_NAMES = [
    "CPR:3",   # Upregulator/Activator
    "CPR:4",   # Downregulator/Inhibitor
    "CPR:5",   # Agonist
    "CPR:6",   # Antagonist
    "CPR:9",   # Substrate/Product
    "false",   # No relation
]

_LABEL_TO_ID = {label: idx for idx, label in enumerate(_LABEL_NAMES)}

# BigBIO KB schema may use different relation type names
_NORMALIZED_LABEL_MAP = {
    # BigBIO naming -> our label names
    "UPREGULATOR": "CPR:3",
    "ACTIVATOR": "CPR:3",
    "DOWNREGULATOR": "CPR:4",
    "INHIBITOR": "CPR:4",
    "AGONIST": "CPR:5",
    "AGONIST-ACTIVATOR": "CPR:5",
    "AGONIST-INHIBITOR": "CPR:5",
    "ANTAGONIST": "CPR:6",
    "SUBSTRATE": "CPR:9",
    "PRODUCT-OF": "CPR:9",
    "SUBSTRATE_PRODUCT-OF": "CPR:9",
}

# Map from split names to HuggingFace split keys
_SPLIT_MAP = {
    "train": "train",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
}


@DATASET_REGISTRY.register("chemprot")
class ChemProtDataset(BaseDataset):
    """ChemProt: BioCreative VI chemical-protein interaction dataset.

    Uses the bigbio/chemprot dataset from HuggingFace.
    """

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

                # Map to label ID, trying both raw type and normalized forms
                label_id = None
                if rel_type in _LABEL_TO_ID:
                    label_id = _LABEL_TO_ID[rel_type]
                elif rel_type in _NORMALIZED_LABEL_MAP:
                    label_id = _LABEL_TO_ID[_NORMALIZED_LABEL_MAP[rel_type]]
                    rel_type = _NORMALIZED_LABEL_MAP[rel_type]
                else:
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
                        label_id=label_id,
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
