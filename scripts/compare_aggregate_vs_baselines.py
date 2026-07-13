#!/usr/bin/env python3
"""Side-by-side comparison: aggregate-model per-source scores vs per-dataset baselines.

The aggregate JSON comes from ``medre-bench evaluate-aggregate`` (per-source
binary metrics). Baseline scores are read from the standard outputs/ tree
(``outputs/<model>__<dataset>__seed<N>/<timestamp>/metrics.json``), which stores
each run's validation-set multi-class metrics. For each source dataset, the
"best baseline" is the model with the highest mean eval_micro_f1 across seeds
(or the model given for that dataset in ``--best-models best_models.yaml``).

Because baselines were trained multi-class and the aggregate model is scored
binary, the two micro-F1 columns are labeled explicitly to avoid misreading.
Baselines whose training used binary_mode=true are already binary and directly
comparable; the script surfaces this via a "mode" column.

Example:
    python scripts/compare_aggregate_vs_baselines.py \\
        --aggregate-eval outputs/aggregate_eval/aggregate_eval_test.json \\
        --baselines-dir outputs/

Optional:
    --best-models best_models.yaml   YAML mapping dataset -> baseline model name
    --output-format {table,csv,latex,markdown}
    --output-file report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

_SOURCE_DATASETS = ["bc5cdr", "biored", "chem_dis_gene", "chemprot", "ddi", "drugprot", "euadr"]


def _load_metrics(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - be tolerant to malformed files
        return None


def _config_snapshot(run_dir: Path) -> dict[str, Any]:
    cfg = run_dir / "config_snapshot.yaml"
    if not cfg.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(cfg.read_text()) or {}
    except Exception:  # noqa: BLE001
        return {}


def _collect_baseline_runs(baselines_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return {dataset: {model: [run_summary, ...]}} scanning outputs/ tree."""
    runs: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for run in baselines_dir.iterdir():
        if not run.is_dir():
            continue
        name = run.name
        if name.count("__") < 2:
            continue
        model, dataset, seed_tag = name.split("__", 2)
        if dataset == "aggregate":
            continue
        for ts in run.iterdir():
            if not ts.is_dir():
                continue
            m = _load_metrics(ts / "metrics.json")
            if not m or "eval" not in m:
                continue
            snap = _config_snapshot(ts)
            binary_mode = bool(snap.get("dataset", {}).get("binary_mode", False))
            eval_metrics = m["eval"]
            runs[dataset][model].append({
                "seed": seed_tag.replace("seed", ""),
                "timestamp": ts.name,
                "binary_mode": binary_mode,
                "micro_f1": eval_metrics.get("eval_micro_f1"),
                "macro_f1": eval_metrics.get("eval_macro_f1"),
                "accuracy": eval_metrics.get("eval_accuracy"),
                "cleaning_strategy": snap.get("dataset", {}).get("cleaning_strategy"),
                "balance_train": snap.get("dataset", {}).get("balance_train"),
            })
    return runs


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return float(statistics.mean(xs)) if xs else float("nan")


def _pick_best_baseline(
    dataset: str,
    per_model_runs: dict[str, list[dict[str, Any]]],
    pinned_model: str | None,
) -> tuple[str | None, dict[str, Any]]:
    """Return (best_model_name, summary) for a dataset. Summary is a dict with
    mean/std/n and the pinned/best mode."""
    if pinned_model:
        model = pinned_model
        if model not in per_model_runs:
            return model, {"available": False}
        runs = per_model_runs[model]
    else:
        ranked = sorted(
            per_model_runs.items(),
            key=lambda kv: _mean([r["micro_f1"] for r in kv[1]]),
            reverse=True,
        )
        if not ranked:
            return None, {"available": False}
        model, runs = ranked[0]

    micro = [r["micro_f1"] for r in runs if r["micro_f1"] is not None]
    macro = [r["macro_f1"] for r in runs if r["macro_f1"] is not None]
    binary_modes = {r["binary_mode"] for r in runs}
    return model, {
        "available": True,
        "n_seeds": len(runs),
        "micro_f1_mean": _mean(micro),
        "micro_f1_std": statistics.stdev(micro) if len(micro) > 1 else 0.0,
        "macro_f1_mean": _mean(macro),
        "macro_f1_std": statistics.stdev(macro) if len(macro) > 1 else 0.0,
        "binary_mode": "binary" if binary_modes == {True} else ("multi" if binary_modes == {False} else "mixed"),
        "cleaning": next((r.get("cleaning_strategy") for r in runs), None),
        "ros": next((r.get("balance_train") for r in runs), None),
    }


def _fmt(x: Any, spec: str = ".4f") -> str:
    if x is None or (isinstance(x, float) and x != x):  # NaN
        return "—"
    if isinstance(x, float):
        return format(x, spec)
    return str(x)


def _render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Simple monospace table. columns is list of (key, header)."""
    widths = {k: max(len(h), max((len(_fmt(r.get(k))) for r in rows), default=0)) for k, h in columns}
    header = "  ".join(h.ljust(widths[k]) for k, h in columns)
    sep = "  ".join("-" * widths[k] for k, _ in columns)
    lines = [header, sep]
    for r in rows:
        lines.append("  ".join(_fmt(r.get(k)).ljust(widths[k]) for k, _ in columns))
    return "\n".join(lines)


def _render_csv(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([h for _, h in columns])
    for r in rows:
        w.writerow([_fmt(r.get(k)) for k, _ in columns])
    return buf.getvalue().rstrip()


def _render_markdown(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(h for _, h in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(_fmt(r.get(k)) for k, _ in columns) + " |"
        for r in rows
    ]
    return "\n".join([header, sep, *body])


def _render_latex(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    col_spec = "l" * len(columns)
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline",
        " & ".join(h for _, h in columns) + r" \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(" & ".join(_fmt(r.get(k)) for k, _ in columns) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregate-eval", required=True, help="Path to aggregate_eval_<split>.json")
    parser.add_argument("--baselines-dir", default="outputs", help="Directory containing per-run subdirs (default: outputs)")
    parser.add_argument("--best-models", default=None, help="Optional YAML: {dataset: model} pinning which baseline model to compare per source")
    parser.add_argument("--sources", default=None, help="Comma-separated subset of sources; default = all present in the aggregate JSON")
    parser.add_argument("--output-format", choices=["table", "csv", "latex", "markdown"], default="table")
    parser.add_argument("--output-file", default=None, help="Write to this path instead of stdout")
    args = parser.parse_args()

    agg_path = Path(args.aggregate_eval).expanduser().resolve()
    if not agg_path.exists():
        print(f"error: aggregate-eval file not found: {agg_path}", file=sys.stderr)
        return 2
    aggregate = json.loads(agg_path.read_text())
    per_source_agg: dict[str, dict[str, Any]] = aggregate.get("per_source", {})

    baselines_dir = Path(args.baselines_dir).expanduser().resolve()
    if not baselines_dir.exists():
        print(f"error: baselines dir not found: {baselines_dir}", file=sys.stderr)
        return 2
    baseline_runs = _collect_baseline_runs(baselines_dir)

    pinned: dict[str, str] = {}
    if args.best_models:
        import yaml

        mapping = yaml.safe_load(Path(args.best_models).read_text()) or {}
        pinned = {str(k): str(v) for k, v in mapping.items()}

    sources = [s.strip() for s in args.sources.split(",")] if args.sources else _SOURCE_DATASETS
    sources = [s for s in sources if s in per_source_agg or s in baseline_runs]

    rows: list[dict[str, Any]] = []
    for src in sources:
        agg = per_source_agg.get(src, {})
        pinned_model = pinned.get(src)
        best_model, base = _pick_best_baseline(src, baseline_runs.get(src, {}), pinned_model)

        rows.append({
            "source": src,
            "n": agg.get("n"),
            "agg_micro_f1": agg.get("micro_f1"),
            "agg_macro_f1": agg.get("macro_f1"),
            "agg_f1_pos": agg.get("f1_positive"),
            "agg_P_pos": agg.get("precision_positive"),
            "agg_R_pos": agg.get("recall_positive"),
            "baseline_model": best_model if base.get("available") else "—",
            "baseline_mode": base.get("binary_mode", "—") if base.get("available") else "—",
            "baseline_seeds": base.get("n_seeds") if base.get("available") else None,
            "baseline_micro_f1": base.get("micro_f1_mean") if base.get("available") else None,
            "baseline_macro_f1": base.get("macro_f1_mean") if base.get("available") else None,
        })

    # Add a pooled row if aggregate produced one.
    combined = aggregate.get("combined") or {}
    if combined:
        rows.append({
            "source": "combined",
            "n": combined.get("n"),
            "agg_micro_f1": combined.get("micro_f1"),
            "agg_macro_f1": combined.get("macro_f1"),
            "agg_f1_pos": combined.get("f1_positive"),
            "agg_P_pos": combined.get("precision_positive"),
            "agg_R_pos": combined.get("recall_positive"),
            "baseline_model": "—",
            "baseline_mode": "—",
            "baseline_seeds": None,
            "baseline_micro_f1": None,
            "baseline_macro_f1": None,
        })

    columns = [
        ("source", "source"),
        ("n", "n"),
        ("agg_micro_f1", "AGG binary micro_f1"),
        ("agg_macro_f1", "AGG binary macro_f1"),
        ("agg_f1_pos", "AGG F1(+)"),
        ("agg_P_pos", "AGG P(+)"),
        ("agg_R_pos", "AGG R(+)"),
        ("baseline_model", "baseline model"),
        ("baseline_mode", "mode"),
        ("baseline_seeds", "n_seeds"),
        ("baseline_micro_f1", "baseline micro_f1"),
        ("baseline_macro_f1", "baseline macro_f1"),
    ]

    if args.output_format == "csv":
        rendered = _render_csv(rows, columns)
    elif args.output_format == "latex":
        rendered = _render_latex(rows, columns)
    elif args.output_format == "markdown":
        rendered = _render_markdown(rows, columns)
    else:
        rendered = _render_table(rows, columns)

    footer_notes = [
        "",
        "Notes:",
        "  AGG columns are BINARY (any relation vs NO_RELATION) from the aggregate model's argmax.",
        "  baseline columns come from each run's metrics.json (validation split during training).",
        "  If mode = 'multi', baseline F1 is multi-class and NOT strictly comparable to AGG binary F1.",
        "  If mode = 'binary', the baseline was trained with binary_mode=true and is directly comparable.",
    ]
    rendered = rendered + "\n" + "\n".join(footer_notes) if args.output_format == "table" else rendered

    if args.output_file:
        Path(args.output_file).write_text(rendered + "\n")
        print(f"wrote {args.output_file}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
