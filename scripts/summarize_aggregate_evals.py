#!/usr/bin/env python3
"""Aggregate per-seed evaluation JSONs into mean +/- std tables per model.

Consumes the two kinds of output produced by ``scripts/evaluate_aggregate_all.sh``:

* ``<run>/<ts>/aggregate_test_eval/metrics_aggregate_<split>.json``
  from ``medre-bench evaluate --dataset aggregate`` (6-class multiclass metrics)
* ``<run>/<ts>/aggregate_eval/aggregate_eval_<split>.json``
  from ``medre-bench evaluate-aggregate`` (per-source binary metrics)

Produces two tables:

1. **Aggregate/<split>** — one row per model, mean +/- std across seeds for
   micro F1, macro F1, weighted F1, accuracy, ROC-AUC (when available).
2. **Per-source/<split>** — for each (model, source) pair, mean +/- std across
   seeds for binary micro F1, macro F1, and positive-class P/R/F1.

Optionally merges with per-dataset baseline numbers if ``--baselines-dir`` is
given (uses the same discovery as ``compare_aggregate_vs_baselines.py``).

Example::

    python scripts/summarize_aggregate_evals.py \\
        --outputs-dir outputs \\
        --split test \\
        --output-file reports/aggregate_summary_test.md \\
        --format markdown
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


_METRIC_KEYS_AGG = [
    ("eval_micro_f1", "micro_f1"),
    ("eval_macro_f1", "macro_f1"),
    ("eval_weighted_f1", "weighted_f1"),
    ("eval_accuracy", "accuracy"),
    ("eval_roc_auc_macro", "roc_auc_macro"),
    ("eval_roc_auc_weighted", "roc_auc_weighted"),
]

_METRIC_KEYS_BIN = [
    "micro_f1", "macro_f1",
    "precision_positive", "recall_positive", "f1_positive",
]


def _mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None and v == v]  # drop None + NaN
    if not vals:
        return float("nan"), 0.0
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _fmt(mean: float, std: float, precision: int = 4) -> str:
    if mean != mean:
        return "—"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def _collect(outputs_dir: Path, split: str) -> tuple[
    dict[str, list[dict]],           # aggregate/<split>: model -> [eval_metrics per seed]
    dict[str, dict[str, list[dict]]], # per-source: model -> source -> [metrics per seed]
]:
    aggregate_test: dict[str, list[dict]] = defaultdict(list)
    per_source: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parts = run_dir.name.split("__")
        if len(parts) < 3 or parts[1] != "aggregate":
            continue
        model = parts[0]

        for ts_dir in sorted(run_dir.iterdir()):
            if not ts_dir.is_dir():
                continue

            agg_json = ts_dir / "aggregate_test_eval" / f"metrics_aggregate_{split}.json"
            if agg_json.exists():
                try:
                    payload = json.loads(agg_json.read_text())
                    # evaluate writes the eval dict flat; some setups nest it under "eval".
                    eval_metrics = payload.get("eval") if isinstance(payload, dict) and "eval" in payload else payload
                    aggregate_test[model].append(eval_metrics)
                except Exception as exc:  # noqa: BLE001
                    print(f"  warn: {agg_json}: {exc}", file=sys.stderr)

            src_json = ts_dir / "aggregate_eval" / f"aggregate_eval_{split}.json"
            if src_json.exists():
                try:
                    payload = json.loads(src_json.read_text())
                    for src, m in (payload.get("per_source") or {}).items():
                        per_source[model][src].append(m)
                except Exception as exc:  # noqa: BLE001
                    print(f"  warn: {src_json}: {exc}", file=sys.stderr)

    return aggregate_test, per_source


def _render_aggregate_table(aggregate_test: dict[str, list[dict]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, seed_results in sorted(aggregate_test.items()):
        row: dict[str, Any] = {"model": model, "n_seeds": len(seed_results)}
        for src_key, out_key in _METRIC_KEYS_AGG:
            values = [r.get(src_key) for r in seed_results if r.get(src_key) is not None]
            m, s = _mean_std(values)
            row[out_key] = _fmt(m, s)
        rows.append(row)
    return rows


def _render_per_source_table(
    per_source: dict[str, dict[str, list[dict]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in sorted(per_source.keys()):
        for src in sorted(per_source[model].keys()):
            seed_results = per_source[model][src]
            row: dict[str, Any] = {
                "model": model,
                "source": src,
                "n_seeds": len(seed_results),
            }
            for k in _METRIC_KEYS_BIN:
                values = [r.get(k) for r in seed_results if r.get(k) is not None]
                m, s = _mean_std(values)
                row[k] = _fmt(m, s)
            rows.append(row)
    return rows


def _merge_baselines(
    rows: list[dict[str, Any]],
    baselines_dir: Path,
    best_models_yaml: Path | None,
) -> list[dict[str, Any]]:
    """Attach the best per-source baseline's mean micro_f1 for context. Uses the
    same discovery scheme as compare_aggregate_vs_baselines.py."""
    import yaml

    pin = {}
    if best_models_yaml and best_models_yaml.exists():
        pin = yaml.safe_load(best_models_yaml.read_text()) or {}

    # Scan baselines_dir/{model}__{dataset}__seed*/{ts}/metrics.json
    per_ds_model_micro: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run_dir in baselines_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name.count("__") < 2:
            continue
        model, dataset, _ = run_dir.name.split("__", 2)
        if dataset == "aggregate":
            continue
        for ts_dir in run_dir.iterdir():
            if not ts_dir.is_dir():
                continue
            m_file = ts_dir / "metrics.json"
            if not m_file.exists():
                continue
            try:
                m = json.loads(m_file.read_text()).get("eval", {})
                if m.get("eval_micro_f1") is not None:
                    per_ds_model_micro[dataset][model].append(m["eval_micro_f1"])
            except Exception:  # noqa: BLE001
                continue

    def _best_baseline(source: str) -> tuple[str, float, float, int] | None:
        pool = per_ds_model_micro.get(source) or {}
        if not pool:
            return None
        if source in pin and pin[source] in pool:
            model_key = pin[source]
        else:
            model_key = max(pool.keys(), key=lambda k: statistics.mean(pool[k]))
        vals = pool[model_key]
        m, s = _mean_std(vals)
        return model_key, m, s, len(vals)

    out: list[dict[str, Any]] = []
    for row in rows:
        src = row.get("source")
        if src is None:
            out.append(row)
            continue
        best = _best_baseline(src)
        if best is None:
            row["baseline_model"] = "—"
            row["baseline_micro_f1"] = "—"
            row["baseline_n_seeds"] = "—"
        else:
            model_key, m, s, n = best
            row["baseline_model"] = model_key
            row["baseline_micro_f1"] = _fmt(m, s)
            row["baseline_n_seeds"] = n
        out.append(row)
    return out


def _render(rows: list[dict[str, Any]], columns: list[tuple[str, str]], fmt: str) -> str:
    if fmt == "csv":
        import csv
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([h for _, h in columns])
        for r in rows:
            w.writerow([r.get(k, "") for k, _ in columns])
        return buf.getvalue().rstrip()
    if fmt == "markdown":
        head = "| " + " | ".join(h for _, h in columns) + " |"
        sep = "|" + "|".join("---" for _ in columns) + "|"
        body = ["| " + " | ".join(str(r.get(k, "")) for k, _ in columns) + " |" for r in rows]
        return "\n".join([head, sep, *body])
    # table
    widths = [max(len(h), max((len(str(r.get(k, ""))) for r in rows), default=0)) for k, h in columns]
    lines = []
    header = "  ".join(h.ljust(widths[i]) for i, (_, h) in enumerate(columns))
    lines.append(header)
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        lines.append("  ".join(str(r.get(k, "")).ljust(widths[i]) for i, (k, _) in enumerate(columns)))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--split", default="test", help="which split's JSONs to summarize (default: test)")
    p.add_argument("--output-file", default=None)
    p.add_argument("--format", choices=["table", "csv", "markdown"], default="table")
    p.add_argument("--baselines-dir", default=None,
                   help="If given, attach the best per-source baseline's micro_f1 to the per-source table")
    p.add_argument("--best-models", default=None,
                   help="Optional YAML pinning baseline model per source (same format as compare_aggregate_vs_baselines.py)")
    args = p.parse_args()

    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    aggregate_test, per_source = _collect(outputs_dir, args.split)

    if not aggregate_test and not per_source:
        print(f"no aggregate evaluation results found under {outputs_dir}", file=sys.stderr)
        print(f"did you run scripts/evaluate_aggregate_all.sh with SPLIT={args.split}?", file=sys.stderr)
        return 2

    agg_rows = _render_aggregate_table(aggregate_test)
    src_rows = _render_per_source_table(per_source)

    if args.baselines_dir:
        best_models_p = Path(args.best_models).expanduser() if args.best_models else None
        src_rows = _merge_baselines(src_rows, Path(args.baselines_dir).expanduser().resolve(), best_models_p)

    parts: list[str] = []

    parts.append(f"=== Aggregate/{args.split} — 6-class multiclass (mean ± std across seeds) ===\n")
    agg_cols = [
        ("model", "model"), ("n_seeds", "n_seeds"),
        ("micro_f1", "micro_f1"), ("macro_f1", "macro_f1"),
        ("weighted_f1", "weighted_f1"), ("accuracy", "accuracy"),
        ("roc_auc_macro", "roc_auc_macro"), ("roc_auc_weighted", "roc_auc_weighted"),
    ]
    parts.append(_render(agg_rows, agg_cols, args.format))
    parts.append("")

    parts.append(f"\n=== Per-source/{args.split} — binary metrics (mean ± std across seeds) ===\n")
    src_cols = [
        ("model", "model"), ("source", "source"), ("n_seeds", "n_seeds"),
        ("micro_f1", "micro_f1"), ("macro_f1", "macro_f1"),
        ("precision_positive", "P(+)"), ("recall_positive", "R(+)"), ("f1_positive", "F1(+)"),
    ]
    if args.baselines_dir:
        src_cols.extend([
            ("baseline_model", "baseline_model"),
            ("baseline_micro_f1", "baseline_micro_f1"),
            ("baseline_n_seeds", "baseline_seeds"),
        ])
    parts.append(_render(src_rows, src_cols, args.format))

    rendered = "\n".join(parts) + "\n"

    if args.output_file:
        Path(args.output_file).write_text(rendered)
        print(f"wrote {args.output_file}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
