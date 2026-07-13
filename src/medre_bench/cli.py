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
def predict(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Path to a saved checkpoint directory (contains model.safetensors and tokenizer files)"),
    text: str = typer.Option(..., "--text", "-t", help="Sentence or multi-sentence text to score"),
    entity1: str = typer.Option(..., "--entity1", "-e1", help="Text of the first entity (case-insensitive substring match)"),
    entity2: str = typer.Option(..., "--entity2", "-e2", help="Text of the second entity"),
    entity1_type: str = typer.Option("ENTITY", "--entity1-type", help="Type label used inside the entity marker for entity1"),
    entity2_type: str = typer.Option("ENTITY", "--entity2-type", help="Type label used inside the entity marker for entity2"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of multi-class labels to display per sentence"),
    output_json: bool = typer.Option(False, "--json", help="Emit the full result as JSON instead of a human-readable summary"),
) -> None:
    """Predict whether text contains a relation between two entities.

    Splits the text into sentences, locates both entities in each, and scores
    every sentence that contains both. Prints a binary RELATION / NO_RELATION
    decision plus the top-k multi-class labels per sentence, and an aggregate
    document-level decision.
    """
    from medre_bench.inference.predictor import run_prediction

    result = run_prediction(
        checkpoint_path=checkpoint,
        text=text,
        entity1=entity1,
        entity2=entity2,
        entity1_type=entity1_type,
        entity2_type=entity2_type,
        top_k=top_k,
    )

    if output_json:
        typer.echo(json.dumps(result, indent=2))
        return

    doc = result["document_prediction"]
    typer.echo(
        f"Model: {result['model_name']} | Dataset: {result['dataset_name']} "
        f"| binary_mode_checkpoint: {result['binary_mode_checkpoint']}"
    )
    typer.echo(f"Entities: {entity1!r} <-> {entity2!r}")
    typer.echo(
        f"Document decision: {doc['binary']} "
        f"(max P(relation)={doc['max_p_relation']:.4f}; "
        f"{doc['n_matched_sentences']}/{doc['n_total_sentences']} sentences matched)"
    )

    if not result["sentence_predictions"]:
        typer.echo("\nNo sentence contained both entities; no scoring performed.")
        return

    for i, p in enumerate(result["sentence_predictions"], 1):
        typer.echo(f"\n--- Sentence {i} ---")
        typer.echo(f"  {p['sentence']}")
        typer.echo(f"  binary: {p['binary']}   P(relation)={p['p_relation']:.4f}")
        typer.echo(f"  top-{len(p['top_k'])} labels:")
        for row in p["top_k"]:
            typer.echo(f"    - {row['label']:<28} {row['prob']:.4f}")


@app.command(name="evaluate-aggregate")
def evaluate_aggregate(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Path to an aggregate-trained checkpoint directory"),
    split: str = typer.Option("test", "--split", "-s", help="Split to evaluate on each source ('validation' or 'test')"),
    sources: Optional[str] = typer.Option(None, "--sources", help="Comma-separated subset of source datasets; default = all 7 sources"),
    batch_size: int = typer.Option(32, "--batch-size", help="Inference batch size"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Directory to save per-source metrics JSON"),
) -> None:
    """Evaluate an aggregate-trained checkpoint on each source dataset's own split.

    Ground truth and prediction are collapsed to binary (any relation vs
    NO_RELATION) so the results are directly comparable to the per-dataset
    baseline models. Reports per-source binary micro/macro F1 + a pooled
    'combined' row over all scored examples.
    """
    from medre_bench.evaluation.aggregate_eval import run_aggregate_evaluation

    source_list = [s.strip() for s in sources.split(",")] if sources else None
    result = run_aggregate_evaluation(
        checkpoint_path=checkpoint,
        split=split,
        sources=source_list,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    header = f"{'source':<18}  {'n':>6}  {'micro_f1':>9}  {'macro_f1':>9}  {'P(+)':>7}  {'R(+)':>7}  {'F1(+)':>7}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for name, m in result["per_source"].items():
        typer.echo(
            f"{name:<18}  {m['n']:>6d}  {m['micro_f1']:>9.4f}  {m['macro_f1']:>9.4f}  "
            f"{m['precision_positive']:>7.4f}  {m['recall_positive']:>7.4f}  {m['f1_positive']:>7.4f}"
        )
    c = result.get("combined") or {}
    if c:
        typer.echo("-" * len(header))
        typer.echo(
            f"{'combined':<18}  {c['n']:>6d}  {c['micro_f1']:>9.4f}  {c['macro_f1']:>9.4f}  "
            f"{c['precision_positive']:>7.4f}  {c['recall_positive']:>7.4f}  {c['f1_positive']:>7.4f}"
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
