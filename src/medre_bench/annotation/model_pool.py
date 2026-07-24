"""Lazy checkpoint loader with a small in-memory cache.

The annotate pipeline needs to run the aggregate model plus up to seven
per-dataset models. Each checkpoint is 300 MB - 1.5 GB of weights and takes
~5 s to load; we amortize that cost by caching a single :class:`LoadedModel`
per key.

Two eviction modes:

* Default: every checkpoint stays loaded once used (fastest for GPU with
  enough VRAM to hold all eight, ~10-12 GB).
* ``single_at_a_time=True``: evict the previous model before loading the
  next one (slower but keeps peak memory to one checkpoint at a time; useful
  on tight GPUs or CPU).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from medre_bench.datasets.preprocessing import BINARY_LABEL_NAMES
from medre_bench.utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class LoadedModel:
    """A fully constructed, ready-to-run checkpoint bundle."""

    key: str  # 'aggregate' / 'bc5cdr' / ...
    tokenizer: Any
    model: torch.nn.Module
    label_names: list[str]
    num_labels: int
    entity_marker_strategy: str
    max_seq_length: int
    model_name: str
    dataset_name: str
    binary_mode: bool
    device: torch.device


class ModelPool:
    """Lazy-load and cache checkpoints keyed by dataset name.

    Keys in the mapping usually match the dataset registry names
    (``aggregate``, ``bc5cdr``, ...). Paths must point at a directory
    containing ``model.safetensors`` (or ``pytorch_model.bin``) and
    tokenizer files, with a ``config_snapshot.yaml`` reachable in one of
    the ancestor directories.
    """

    def __init__(
        self,
        paths: dict[str, str],
        device_preference: str = "auto",
        single_at_a_time: bool = False,
    ) -> None:
        self._paths = {k: Path(v).expanduser() for k, v in paths.items()}
        self._cache: dict[str, LoadedModel] = {}
        self._single = single_at_a_time
        self._device = self._resolve_device(device_preference)

    @staticmethod
    def _resolve_device(pref: str) -> torch.device:
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device='cuda' but torch.cuda.is_available() is False")
            return torch.device("cuda")
        # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def get(self, key: str) -> LoadedModel:
        if key in self._cache:
            return self._cache[key]
        if key not in self._paths:
            raise KeyError(
                f"No checkpoint path configured for {key!r} in best_models.yaml"
            )
        if self._single and self._cache:
            self._evict_all()
        loaded = self._load(key, self._paths[key])
        self._cache[key] = loaded
        return loaded

    def _evict_all(self) -> None:
        for key, loaded in list(self._cache.items()):
            loaded.model.to("cpu")
            del loaded.model
            del loaded.tokenizer
            self._cache.pop(key)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load(self, key: str, ckpt_dir: Path) -> LoadedModel:
        from transformers import AutoTokenizer

        import medre_bench.datasets  # noqa: F401 - registration side-effects
        import medre_bench.models  # noqa: F401
        from medre_bench.models.base import get_entity_marker_tokens
        from medre_bench.registry import DATASET_REGISTRY, MODEL_REGISTRY
        from medre_bench.training.trainer import REModel

        # Locate config_snapshot.yaml the same way evaluator/predictor do.
        candidates = [
            ckpt_dir / "config_snapshot.yaml",
            ckpt_dir.parent / "config_snapshot.yaml",
            ckpt_dir.parent.parent / "config_snapshot.yaml",
            ckpt_dir.parent.parent.parent / "config_snapshot.yaml",
        ]
        config_path = next((p for p in candidates if p.exists()), None)
        if config_path is None:
            raise FileNotFoundError(
                f"Cannot find config_snapshot.yaml near {ckpt_dir}"
            )
        saved = yaml.safe_load(config_path.read_text())
        model_cfg = saved.get("model", {})
        dataset_cfg = saved.get("dataset", {})
        entity_marker_strategy = model_cfg.get(
            "entity_marker_strategy", "typed_entity_marker_punct"
        )
        max_seq_length = int(model_cfg.get("max_seq_length", 512))
        model_name = model_cfg.get("name", "bert-base")
        dataset_name = dataset_cfg.get("name")
        binary_mode = bool(dataset_cfg.get("binary_mode", False))

        # Recover label set
        if binary_mode:
            num_labels = 2
            label_names = list(BINARY_LABEL_NAMES)
        else:
            if not dataset_name:
                raise ValueError(
                    f"{ckpt_dir}: config_snapshot.yaml missing dataset.name and "
                    "not binary_mode; cannot recover label set."
                )
            ds_cls = DATASET_REGISTRY.get(dataset_name)
            ds = ds_cls()
            num_labels = ds.num_labels()
            label_names = ds.label_names()

        # Tokenizer with fallback: some tokenizer.json files were saved by
        # a newer tokenizers than what's installed. The base pretrained
        # tokenizer is safe for typed_entity_marker_punct / typed_entity_marker
        # strategies (no new special tokens were added).
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
        except Exception as exc:  # noqa: BLE001
            base_probe = MODEL_REGISTRY.get(model_name)()
            base_pretrained = base_probe.pretrained_model_name()
            logger.warning(
                f"{key}: tokenizer load from checkpoint failed "
                f"({type(exc).__name__}); using base pretrained {base_pretrained!r}"
            )
            tokenizer = AutoTokenizer.from_pretrained(base_pretrained)

        base_cls = MODEL_REGISTRY.get(model_name)
        base = base_cls()
        marker_tokens = get_entity_marker_tokens(entity_marker_strategy)
        base.build(
            num_labels=num_labels,
            entity_marker_tokens=marker_tokens if marker_tokens else None,
        )
        base.tokenizer = tokenizer
        model = REModel(base, num_labels=num_labels)

        sf = ckpt_dir / "model.safetensors"
        pb = ckpt_dir / "pytorch_model.bin"
        if sf.exists():
            from safetensors.torch import load_file

            state = load_file(str(sf), device=str(self._device))
        elif pb.exists():
            state = torch.load(pb, map_location=self._device, weights_only=True)
        else:
            raise FileNotFoundError(f"No model weights in {ckpt_dir}")
        model.load_state_dict(state, strict=False)
        model.to(self._device).eval()

        logger.info(
            f"loaded {key}: model={model_name}, dataset={dataset_name}, "
            f"labels={num_labels}, binary_mode={binary_mode}, device={self._device}"
        )
        return LoadedModel(
            key=key,
            tokenizer=tokenizer,
            model=model,
            label_names=label_names,
            num_labels=num_labels,
            entity_marker_strategy=entity_marker_strategy,
            max_seq_length=max_seq_length,
            model_name=model_name,
            dataset_name=dataset_name or key,
            binary_mode=binary_mode,
            device=self._device,
        )
