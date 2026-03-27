"""Configuration loading and merging."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from medre_bench.config.schema import ExperimentConfig


def _find_configs_dir() -> Path:
    """Find the configs directory relative to the package."""
    # Walk up from this file to find the project root containing configs/
    current = Path(__file__).resolve()
    for parent in current.parents:
        configs_dir = parent / "configs"
        if configs_dir.is_dir():
            return configs_dir
    raise FileNotFoundError("Could not find configs/ directory")


def load_config(
    model: str,
    dataset: str,
    config_path: Optional[str] = None,
    overrides: Optional[dict] = None,
    configs_dir: Optional[str] = None,
) -> ExperimentConfig:
    """Load and merge configuration from YAML files and CLI overrides.

    Resolution order:
    1. configs/defaults.yaml
    2. configs/datasets/<dataset>.yaml
    3. configs/models/<model>.yaml
    4. Experiment config file (if provided)
    5. CLI overrides dict
    """
    cfg_dir = Path(configs_dir) if configs_dir else _find_configs_dir()

    # Start with defaults
    defaults_path = cfg_dir / "defaults.yaml"
    if defaults_path.exists():
        base = OmegaConf.load(defaults_path)
    else:
        base = OmegaConf.create({})

    # Merge dataset config
    dataset_cfg_path = cfg_dir / "datasets" / f"{dataset}.yaml"
    if dataset_cfg_path.exists():
        dataset_cfg = OmegaConf.load(dataset_cfg_path)
        base = OmegaConf.merge(base, dataset_cfg)

    # Merge model config
    model_cfg_path = cfg_dir / "models" / f"{model}.yaml"
    if model_cfg_path.exists():
        model_cfg = OmegaConf.load(model_cfg_path)
        base = OmegaConf.merge(base, model_cfg)

    # Merge experiment config
    if config_path:
        exp_cfg = OmegaConf.load(config_path)
        base = OmegaConf.merge(base, exp_cfg)

    # Ensure model and dataset names are set
    base_dict = OmegaConf.to_container(base, resolve=True)
    if not isinstance(base_dict, dict):
        base_dict = {}

    base_dict.setdefault("model", {})
    base_dict.setdefault("dataset", {})
    base_dict["model"]["name"] = model
    base_dict["dataset"]["name"] = dataset
    base_dict.setdefault("experiment_name", f"{model}_{dataset}")

    # Apply overrides
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        base_dict = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(base_dict), override_cfg),
            resolve=True,
        )

    return ExperimentConfig(**base_dict)
