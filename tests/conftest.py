"""Shared test fixtures."""

import pytest


@pytest.fixture
def sample_text():
    return "Aspirin inhibits COX-2 enzyme activity in patients."


@pytest.fixture
def sample_config_dict():
    return {
        "experiment_name": "test_run",
        "output_dir": "/tmp/medre-bench-test",
        "model": {
            "name": "bert-base",
            "max_seq_length": 128,
            "entity_marker_strategy": "typed_entity_marker_punct",
        },
        "dataset": {
            "name": "chemprot",
        },
        "training": {
            "seed": 42,
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 2e-5,
        },
        "logging": {
            "use_wandb": False,
            "use_tensorboard": False,
        },
    }
