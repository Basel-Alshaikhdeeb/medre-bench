"""EU-ADR dataset adapter for drug-disease-target relation extraction."""

from __future__ import annotations

import importlib.util
import logging
import sys

from medre_bench.datasets.base import BaseDataset, RelationExample
from medre_bench.registry import DATASET_REGISTRY

logger = logging.getLogger(__name__)

_LABEL_NAMES = [
    "PA",  # Positive Association
    "SA",  # Speculative Association
    "NA",  # Negative Association
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

        if split in ("test",):
            return all_examples[:test_size]
        if split in ("validation", "dev"):
            return all_examples[test_size : test_size + val_size]
        if split == "train":
            return all_examples[test_size + val_size :]

        raise ValueError(f"Unknown split: {split}")

    def _load_all(self) -> list[RelationExample]:
        """Load EU-ADR by downloading and executing the dataset script directly."""
        from huggingface_hub import hf_hub_download

        # Download the dataset script and its dependency
        script_path = hf_hub_download("bigbio/euadr", "euadr.py", repo_type="dataset")
        hub_path = hf_hub_download("bigbio/euadr", "bigbiohub.py", repo_type="dataset")

        # Create a fake package so relative imports (from .bigbiohub) work
        import types
        from pathlib import Path

        pkg_dir = str(Path(script_path).parent)
        pkg_name = "_euadr_pkg"

        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [pkg_dir]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

        # Load bigbiohub as a submodule of the fake package
        hub_spec = importlib.util.spec_from_file_location(f"{pkg_name}.bigbiohub", hub_path)
        hub_mod = importlib.util.module_from_spec(hub_spec)
        sys.modules[f"{pkg_name}.bigbiohub"] = hub_mod
        hub_spec.loader.exec_module(hub_mod)

        # Load the dataset script as a submodule
        ds_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.euadr", script_path, submodule_search_locations=[]
        )
        loader_mod = importlib.util.module_from_spec(ds_spec)
        loader_mod.__package__ = pkg_name
        sys.modules[f"{pkg_name}.euadr"] = loader_mod
        ds_spec.loader.exec_module(loader_mod)

        # Find the builder class (subclass of datasets.GeneratorBasedBuilder)
        import datasets

        builder_cls = None
        for attr_name in dir(loader_mod):
            attr = getattr(loader_mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, datasets.GeneratorBasedBuilder)
                and attr is not datasets.GeneratorBasedBuilder
            ):
                builder_cls = attr
                break

        if builder_cls is None:
            raise RuntimeError("Could not find dataset builder class in euadr.py")

        # Find the bigbio_kb config
        builder = None
        for config in builder_cls.BUILDER_CONFIGS:
            if "bigbio_kb" in config.name:
                builder = builder_cls(config_name=config.name)
                break

        if builder is None:
            raise RuntimeError(
                f"Could not find bigbio_kb config. Available: "
                f"{[c.name for c in builder_cls.BUILDER_CONFIGS]}"
            )

        builder.download_and_prepare()
        ds = builder.as_dataset(split="train")

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
                f"No examples loaded from bigbio/euadr. "
                f"Relation types found in data: {seen_rel_types}. "
                f"Expected types: {set(_LABEL_TO_ID.keys())}"
            )

        return examples
