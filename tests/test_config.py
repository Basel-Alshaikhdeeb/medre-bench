"""Tests for configuration schema and loading."""

import pytest
from pydantic import ValidationError

from medre_bench.config.schema import (
    DatasetConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    TrainingConfig,
)


def test_model_config_defaults():
    cfg = ModelConfig(name="bert-base")
    assert cfg.max_seq_length == 512
    assert cfg.entity_marker_strategy == "typed_entity_marker_punct"


def test_model_config_requires_name():
    with pytest.raises(ValidationError):
        ModelConfig()


def test_dataset_config():
    cfg = DatasetConfig(name="chemprot")
    assert cfg.max_train_samples is None


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.seed == 42
    assert cfg.epochs == 10
    assert cfg.learning_rate == 2e-5
    assert cfg.fp16 is True


def test_experiment_config(sample_config_dict):
    cfg = ExperimentConfig(**sample_config_dict)
    assert cfg.experiment_name == "test_run"
    assert cfg.model.name == "bert-base"
    assert cfg.dataset.name == "chemprot"
    assert cfg.training.epochs == 1
    assert cfg.logging.use_wandb is False


def test_experiment_config_requires_names():
    with pytest.raises(ValidationError):
        ExperimentConfig(experiment_name="test")
