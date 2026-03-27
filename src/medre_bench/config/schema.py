"""Pydantic configuration schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model-specific configuration."""

    name: str = Field(..., description="Registry key, e.g. 'pubmedbert'")
    max_seq_length: int = Field(512, description="Maximum input sequence length")
    entity_marker_strategy: str = Field(
        "typed_entity_marker_punct",
        description="Entity marking strategy: 'typed_entity_marker_punct', 'typed_entity_marker', 'standard'",
    )


class DatasetConfig(BaseModel):
    """Dataset-specific configuration."""

    name: str = Field(..., description="Registry key, e.g. 'chemprot'")
    max_train_samples: Optional[int] = Field(None, description="Limit training samples (for debugging)")
    max_eval_samples: Optional[int] = Field(None, description="Limit eval samples (for debugging)")


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    seed: int = 42
    epochs: int = 10
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 3
    metric_for_best_model: str = "micro_f1"
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 4


class LoggingConfig(BaseModel):
    """Logging and experiment tracking configuration."""

    use_wandb: bool = True
    use_tensorboard: bool = True
    wandb_project: str = "medre-bench"
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 50
    eval_steps: Optional[int] = None
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    experiment_name: str = Field(..., description="Name for this experiment run")
    output_dir: str = Field("outputs", description="Root output directory")
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig = TrainingConfig()
    logging: LoggingConfig = LoggingConfig()
