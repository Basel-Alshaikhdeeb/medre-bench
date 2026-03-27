"""Custom callbacks for training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class ConfigSnapshotCallback(TrainerCallback):
    """Save config snapshot at the start of training."""

    def __init__(self, config_dict: dict[str, Any], run_dir: Path):
        self.config_dict = config_dict
        self.run_dir = run_dir

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        from medre_bench.utils.io import save_config_snapshot, save_environment_info

        save_config_snapshot(self.config_dict, self.run_dir)
        save_environment_info(self.run_dir)


class WandbExtrasCallback(TrainerCallback):
    """Log extra artifacts to W&B at end of training."""

    def __init__(self, label_names: list[str]):
        self.label_names = label_names

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import wandb

            if wandb.run is None:
                return

            if metrics:
                # Log per-class F1 if available
                per_class_metrics = {
                    k: v for k, v in metrics.items() if k.startswith("eval_class_")
                }
                if per_class_metrics:
                    table_data = [
                        [k.replace("eval_class_", "").replace("_f1", ""), v]
                        for k, v in per_class_metrics.items()
                        if k.endswith("_f1")
                    ]
                    if table_data:
                        table = wandb.Table(data=table_data, columns=["class", "f1"])
                        wandb.log(
                            {
                                "per_class_f1": wandb.plot.bar(
                                    table, "class", "f1", title="Per-Class F1"
                                )
                            }
                        )
        except ImportError:
            pass
