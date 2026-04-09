"""Training pipeline for relation extraction models."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from medre_bench.config.schema import ExperimentConfig
from medre_bench.datasets.base import BaseDataset, RelationExample, apply_entity_markers
from medre_bench.models.base import BaseREModel, get_entity_marker_tokens
from medre_bench.registry import DATASET_REGISTRY, MODEL_REGISTRY
from medre_bench.training.callbacks import ConfigSnapshotCallback, WandbExtrasCallback
from medre_bench.training.metrics import compute_metrics
from medre_bench.utils.io import create_run_dir, save_metrics
from medre_bench.utils.logging import setup_logger
from medre_bench.utils.seed import seed_everything

logger = setup_logger(__name__)


class RETokenizedDataset(TorchDataset):
    """Tokenized relation extraction dataset for PyTorch DataLoader."""

    def __init__(
        self,
        examples: list[RelationExample],
        tokenizer: Any,
        max_seq_length: int,
        entity_marker_strategy: str,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.entity_marker_strategy = entity_marker_strategy
        self._encodings = self._tokenize_all()

    def _tokenize_all(self) -> dict[str, list]:
        texts = []
        labels = []

        for ex in self.examples:
            # Skip entity markers for sentence-level datasets (no entity offsets)
            has_entities = (ex.entity1_start != ex.entity1_end) or (
                ex.entity2_start != ex.entity2_end
            )
            if has_entities and self.entity_marker_strategy != "none":
                marked_text = apply_entity_markers(
                    text=ex.text,
                    e1_start=ex.entity1_start,
                    e1_end=ex.entity1_end,
                    e1_type=ex.entity1_type,
                    e2_start=ex.entity2_start,
                    e2_end=ex.entity2_end,
                    e2_type=ex.entity2_type,
                    strategy=self.entity_marker_strategy,
                )
            else:
                marked_text = ex.text
            texts.append(marked_text)
            labels.append(ex.label_id)

        encodings = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encodings["labels"] = torch.tensor(labels, dtype=torch.long)
        return encodings

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: val[idx] for key, val in self._encodings.items()}


class REModel(torch.nn.Module):
    """Wrapper that adapts BaseREModel to HuggingFace Trainer interface."""

    def __init__(self, base_model: BaseREModel, num_labels: int):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.config = base_model.encoder.config

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        sd = super().state_dict(*args, **kwargs)
        return {k: v.contiguous() if not v.is_contiguous() else v for k, v in sd.items()}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        result = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # Return a namedtuple-like object compatible with Trainer
        from transformers.modeling_outputs import SequenceClassifierOutput

        return SequenceClassifierOutput(
            loss=result.get("loss"),
            logits=result["logits"],
        )


def run_training(cfg: ExperimentConfig) -> dict[str, Any]:
    """Execute a full training run.

    Args:
        cfg: Fully resolved experiment configuration.

    Returns:
        Dict of final evaluation metrics.
    """
    seed_everything(cfg.training.seed)
    logger.info(f"Starting experiment: {cfg.experiment_name}")

    # Create run directory
    run_dir = create_run_dir(cfg.output_dir, cfg.experiment_name)
    logger.info(f"Run directory: {run_dir}")

    # Load dataset
    import medre_bench.datasets  # noqa: F401 - trigger registration
    dataset_cls = DATASET_REGISTRY.get(cfg.dataset.name)
    dataset: BaseDataset = dataset_cls()
    logger.info(f"Dataset: {dataset.name()} ({dataset.num_labels()} labels)")

    # Load model
    import medre_bench.models  # noqa: F401 - trigger registration
    model_cls = MODEL_REGISTRY.get(cfg.model.name)
    base_model: BaseREModel = model_cls()

    # Build model with entity markers
    entity_marker_tokens = get_entity_marker_tokens(cfg.model.entity_marker_strategy)
    base_model.build(
        num_labels=dataset.num_labels(),
        entity_marker_tokens=entity_marker_tokens if entity_marker_tokens else None,
    )
    logger.info(f"Model: {cfg.model.name} ({base_model.pretrained_model_name()})")

    # Wrap for HF Trainer compatibility
    model = REModel(base_model, num_labels=dataset.num_labels())

    # Load and tokenize data
    train_examples = dataset.load_split("train")
    eval_examples = dataset.load_split("validation")

    if cfg.dataset.max_train_samples:
        train_examples = train_examples[: cfg.dataset.max_train_samples]
    if cfg.dataset.max_eval_samples:
        eval_examples = eval_examples[: cfg.dataset.max_eval_samples]

    logger.info(f"Train examples: {len(train_examples)}, Eval examples: {len(eval_examples)}")

    train_dataset = RETokenizedDataset(
        train_examples,
        base_model.tokenizer,
        cfg.model.max_seq_length,
        cfg.model.entity_marker_strategy,
    )
    eval_dataset = RETokenizedDataset(
        eval_examples,
        base_model.tokenizer,
        cfg.model.max_seq_length,
        cfg.model.entity_marker_strategy,
    )

    # Configure reporting
    report_to = []
    if cfg.logging.use_tensorboard:
        report_to.append("tensorboard")
    if cfg.logging.use_wandb:
        report_to.append("wandb")
        os.environ["WANDB_PROJECT"] = cfg.logging.wandb_project
        if cfg.logging.wandb_entity:
            os.environ["WANDB_ENTITY"] = cfg.logging.wandb_entity

    # Training arguments
    save_ckpt = cfg.training.save_checkpoints
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.eval_batch_size,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        fp16=cfg.training.fp16,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        eval_strategy=cfg.logging.eval_strategy,
        save_strategy=cfg.logging.save_strategy if save_ckpt else "no",
        save_total_limit=cfg.logging.save_total_limit if save_ckpt else 0,
        load_best_model_at_end=save_ckpt,
        metric_for_best_model=cfg.training.metric_for_best_model if save_ckpt else None,
        greater_is_better=True if save_ckpt else None,
        logging_steps=cfg.logging.log_every_n_steps,
        logging_dir=str(run_dir / "tensorboard"),
        report_to=report_to,
        run_name=cfg.experiment_name,
        seed=cfg.training.seed,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        remove_unused_columns=False,
        deepspeed=cfg.training.deepspeed,
    )

    # Callbacks
    callbacks = [
        ConfigSnapshotCallback(cfg.model_dump(), run_dir),
    ]
    if save_ckpt:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.training.early_stopping_patience))
    if cfg.logging.use_wandb:
        callbacks.append(WandbExtrasCallback(label_names=dataset.label_names()))

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Evaluate
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()

    # Save metrics
    all_metrics = {
        "train": {k: v for k, v in train_result.metrics.items()},
        "eval": eval_metrics,
        "config": {
            "model": cfg.model.name,
            "dataset": cfg.dataset.name,
            "seed": cfg.training.seed,
        },
    }
    save_metrics(all_metrics, run_dir)
    logger.info(f"Metrics saved to {run_dir / 'metrics.json'}")

    # Save best model
    trainer.save_model(str(run_dir / "checkpoints" / "best"))
    base_model.tokenizer.save_pretrained(str(run_dir / "checkpoints" / "best"))
    logger.info(f"Best model saved to {run_dir / 'checkpoints' / 'best'}")

    return eval_metrics
