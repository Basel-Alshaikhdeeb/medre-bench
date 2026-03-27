"""Results analysis and comparison utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from tabulate import tabulate

from medre_bench.utils.logging import setup_logger

logger = setup_logger(__name__)


def _collect_results(results_dir: str) -> list[dict[str, Any]]:
    """Collect all metrics.json files from the results directory tree."""
    results = []
    root = Path(results_dir)

    for metrics_file in sorted(root.rglob("metrics.json")):
        with open(metrics_file) as f:
            data = json.load(f)

        config = data.get("config", {})
        eval_metrics = data.get("eval", {})

        results.append({
            "model": config.get("model", "unknown"),
            "dataset": config.get("dataset", "unknown"),
            "seed": config.get("seed", "?"),
            "micro_f1": eval_metrics.get("eval_micro_f1", eval_metrics.get("micro_f1", 0.0)),
            "macro_f1": eval_metrics.get("eval_macro_f1", eval_metrics.get("macro_f1", 0.0)),
            "accuracy": eval_metrics.get("eval_accuracy", eval_metrics.get("accuracy", 0.0)),
            "run_dir": str(metrics_file.parent),
        })

    return results


def _build_comparison_table(results: list[dict[str, Any]]) -> tuple[list[list], list[str]]:
    """Build a model x dataset comparison table with mean and std across seeds."""
    import numpy as np

    # Group by (model, dataset)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        key = (r["model"], r["dataset"])
        grouped.setdefault(key, []).append(r)

    # Get unique models and datasets
    models = sorted(set(r["model"] for r in results))
    datasets = sorted(set(r["dataset"] for r in results))

    headers = ["Model"] + datasets
    rows = []

    for model in models:
        row = [model]
        for dataset in datasets:
            key = (model, dataset)
            if key in grouped:
                f1_scores = [r["micro_f1"] for r in grouped[key]]
                mean = np.mean(f1_scores)
                if len(f1_scores) > 1:
                    std = np.std(f1_scores)
                    row.append(f"{mean:.4f} +/- {std:.4f}")
                else:
                    row.append(f"{mean:.4f}")
            else:
                row.append("-")
        rows.append(row)

    return rows, headers


def compare_results(
    results_dir: str,
    output_format: str = "table",
    output_file: Optional[str] = None,
) -> None:
    """Compare results across all experiments.

    Args:
        results_dir: Root directory containing experiment outputs.
        output_format: One of 'table', 'csv', 'latex'.
        output_file: Optional file path to save output.
    """
    results = _collect_results(results_dir)

    if not results:
        logger.warning(f"No results found in {results_dir}")
        return

    rows, headers = _build_comparison_table(results)

    if output_format == "table":
        table_str = tabulate(rows, headers=headers, tablefmt="grid")
    elif output_format == "csv":
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
        table_str = output.getvalue()
    elif output_format == "latex":
        table_str = tabulate(rows, headers=headers, tablefmt="latex_booktabs")
    else:
        raise ValueError(f"Unknown output format: {output_format}")

    print(table_str)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(table_str)
        logger.info(f"Results saved to {output_file}")

    # Also print detailed per-run results
    print("\n\nDetailed results:")
    detail_rows = []
    for r in results:
        detail_rows.append([
            r["model"],
            r["dataset"],
            r["seed"],
            f"{r['micro_f1']:.4f}",
            f"{r['macro_f1']:.4f}",
            f"{r['accuracy']:.4f}",
        ])

    detail_headers = ["Model", "Dataset", "Seed", "Micro F1", "Macro F1", "Accuracy"]
    print(tabulate(detail_rows, headers=detail_headers, tablefmt="grid"))
