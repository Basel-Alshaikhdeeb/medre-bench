"""CLI entry point for medre-bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="medre-bench",
    help="Medical Relation Extraction Benchmarking Framework",
    no_args_is_help=True,
)


def _parse_overrides(overrides: list[str]) -> dict:
    """Parse dot-notation CLI overrides into nested dict.

    Example: ['training.learning_rate=3e-5', 'training.epochs=15']
    -> {'training': {'learning_rate': 3e-5, 'epochs': 15}}
    """
    result = {}
    for item in overrides:
        if "=" not in item:
            raise typer.BadParameter(f"Override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        # Try to parse as JSON for booleans, numbers, etc.
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass  # Keep as string

        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result


@app.command()
def train(
    model: str = typer.Option(..., "--model", "-m", help="Model registry key"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset registry key"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to experiment YAML config"),
    output_dir: str = typer.Option("outputs", "--output-dir", "-o", help="Output directory"),
    overrides: Optional[list[str]] = typer.Argument(None, help="Config overrides in key=value format"),
) -> None:
    """Train a model on a dataset."""
    from medre_bench.config.loader import load_config
    from medre_bench.training.trainer import run_training

    cli_overrides = _parse_overrides(overrides or [])
    cli_overrides["output_dir"] = output_dir

    cfg = load_config(model=model, dataset=dataset, config_path=config, overrides=cli_overrides)
    run_training(cfg)


@app.command()
def evaluate(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Path to model checkpoint"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset registry key"),
    split: str = typer.Option("test", "--split", "-s", help="Dataset split to evaluate"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory for predictions"),
    overrides: Optional[list[str]] = typer.Argument(None, help="Config overrides"),
) -> None:
    """Evaluate a trained checkpoint on a dataset split."""
    from medre_bench.evaluation.evaluator import run_evaluation

    cli_overrides = _parse_overrides(overrides or [])
    run_evaluation(
        checkpoint_path=checkpoint,
        dataset_name=dataset,
        split=split,
        output_dir=output_dir,
        overrides=cli_overrides,
    )


@app.command()
def sweep(
    models: str = typer.Option(..., "--models", help="Comma-separated model registry keys"),
    datasets: str = typer.Option(..., "--datasets", help="Comma-separated dataset registry keys"),
    seeds: str = typer.Option("42", "--seeds", help="Comma-separated random seeds"),
    output_dir: str = typer.Option("outputs", "--output-dir", "-o", help="Output directory"),
    overrides: Optional[list[str]] = typer.Argument(None, help="Config overrides"),
) -> None:
    """Run training sweep over models x datasets x seeds."""
    from medre_bench.config.loader import load_config
    from medre_bench.training.trainer import run_training

    model_list = [m.strip() for m in models.split(",")]
    dataset_list = [d.strip() for d in datasets.split(",")]
    seed_list = [int(s.strip()) for s in seeds.split(",")]
    cli_overrides = _parse_overrides(overrides or [])

    total = len(model_list) * len(dataset_list) * len(seed_list)
    run_idx = 0

    for model_name in model_list:
        for dataset_name in dataset_list:
            for seed in seed_list:
                run_idx += 1
                typer.echo(f"\n{'='*60}")
                typer.echo(f"Run {run_idx}/{total}: {model_name} x {dataset_name} x seed={seed}")
                typer.echo(f"{'='*60}\n")

                run_overrides = {**cli_overrides, "training": {"seed": seed}}
                run_overrides["output_dir"] = output_dir

                cfg = load_config(
                    model=model_name,
                    dataset=dataset_name,
                    overrides=run_overrides,
                )
                run_training(cfg)


@app.command()
def compare(
    results_dir: str = typer.Option("outputs", "--results-dir", "-r", help="Root results directory"),
    output_format: str = typer.Option("table", "--output-format", "-f", help="Output format: table, csv, latex"),
    output_file: Optional[str] = typer.Option(None, "--output-file", help="Save to file"),
) -> None:
    """Compare results across experiments."""
    from medre_bench.evaluation.analysis import compare_results

    compare_results(
        results_dir=results_dir,
        output_format=output_format,
        output_file=output_file,
    )


@app.command(name="list-models")
def list_models() -> None:
    """List all registered models."""
    import medre_bench.models  # noqa: F401 - triggers auto-registration
    from medre_bench.registry import MODEL_REGISTRY

    models = MODEL_REGISTRY.list_available()
    if not models:
        typer.echo("No models registered.")
        return
    typer.echo("Available models:")
    for m in models:
        typer.echo(f"  - {m}")


@app.command(name="list-datasets")
def list_datasets() -> None:
    """List all registered datasets."""
    import medre_bench.datasets  # noqa: F401 - triggers auto-registration
    from medre_bench.registry import DATASET_REGISTRY

    datasets = DATASET_REGISTRY.list_available()
    if not datasets:
        typer.echo("No datasets registered.")
        return
    typer.echo("Available datasets:")
    for d in datasets:
        typer.echo(f"  - {d}")


if __name__ == "__main__":
    app()
