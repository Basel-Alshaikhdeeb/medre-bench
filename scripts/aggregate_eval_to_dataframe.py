#!/usr/bin/env python3
"""Collapse every ``aggregate_eval_<split>.json`` under an outputs tree into
a pandas view.

Two views:

* ``--view pivot`` (default) — ``model x dataset`` table with the requested
  metric averaged across seeds. One value per cell.
* ``--view raw`` — long-format table with one row per ``(model, dataset, seed)``
  triple and one column per requested metric. Nothing is aggregated.

Discovery walks ``<OUTPUTS_DIR>/*__<DATASET_TAG>__seed*/<ts>/aggregate_eval/aggregate_eval_<split>.json``
where DATASET_TAG defaults to ``aggregate_binary``.

Examples::

    # Mean F1(+) across seeds, model x dataset pivot (default)
    python scripts/aggregate_eval_to_dataframe.py

    # Raw view: one row per (model, dataset, seed) with micro_f1, macro_f1, accuracy
    python scripts/aggregate_eval_to_dataframe.py --view raw

    # Raw view with custom columns, written to CSV
    python scripts/aggregate_eval_to_dataframe.py --view raw \\
        --metrics f1_positive,precision_positive,recall_positive \\
        --output-file raw_pos_metrics.csv

    # Both mean and std side-by-side (pivot only)
    python scripts/aggregate_eval_to_dataframe.py --include-std
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


_AVAILABLE_METRICS = [
    "f1_positive", "precision_positive", "recall_positive",
    "micro_f1", "macro_f1", "accuracy",
    "n", "n_positive_true", "n_positive_pred",
]


def _iter_run_jsons(outputs_dir: Path, dataset_tag: str, split: str):
    """Yield (model, seed, payload_dict) for every eval JSON discovered."""
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parts = run_dir.name.split("__")
        if len(parts) < 3 or parts[1] != dataset_tag:
            continue
        model = parts[0]
        seed = parts[2].replace("seed", "").split("__")[0]

        for ts_dir in sorted(run_dir.iterdir()):
            if not ts_dir.is_dir() or ts_dir.name.startswith("_"):
                continue
            json_path = ts_dir / "aggregate_eval" / f"aggregate_eval_{split}.json"
            if not json_path.exists():
                continue
            try:
                payload = json.loads(json_path.read_text())
            except Exception as exc:  # noqa: BLE001
                print(f"warn: {json_path}: {exc}", file=sys.stderr)
                continue
            yield model, seed, payload


def collect(
    outputs_dir: Path,
    dataset_tag: str,
    split: str,
    metric: str,
    datasets: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (mean, std, n_seeds) DataFrames of shape (models x sources).

    If ``datasets`` is given, only ``per_source`` keys in that set are kept.
    """
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for model, seed, payload in _iter_run_jsons(outputs_dir, dataset_tag, split):
        per_source = payload.get("per_source") or {}
        for source, m in per_source.items():
            if datasets is not None and source not in datasets:
                continue
            v = m.get(metric)
            if v is None:
                continue
            values[model][source].append(float(v))

    if not values:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    models = sorted(values.keys())
    sources = sorted({s for by_src in values.values() for s in by_src.keys()})

    mean_df = pd.DataFrame(index=models, columns=sources, dtype=float)
    std_df = pd.DataFrame(index=models, columns=sources, dtype=float)
    n_df = pd.DataFrame(index=models, columns=sources, dtype="Int64")

    for model in models:
        for source in sources:
            vals = values[model].get(source, [])
            if not vals:
                continue
            mean_df.at[model, source] = float(np.mean(vals))
            std_df.at[model, source] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            n_df.at[model, source] = len(vals)

    mean_df.index.name = "model"
    std_df.index.name = "model"
    n_df.index.name = "model"
    return mean_df, std_df, n_df


def collect_raw(
    outputs_dir: Path,
    dataset_tag: str,
    split: str,
    metrics: list[str],
    datasets: set[str] | None = None,
) -> pd.DataFrame:
    """Return a long-format DataFrame with one row per (model, dataset, seed).

    Columns are ``["model", "dataset", "seed"] + metrics``. Nothing is
    averaged — every JSON contributes as many rows as it has sources under
    ``per_source``. Missing metric values become ``NaN``. If ``datasets`` is
    given, only ``per_source`` keys in that set are kept.
    """
    rows: list[dict] = []
    for model, seed, payload in _iter_run_jsons(outputs_dir, dataset_tag, split):
        per_source = payload.get("per_source") or {}
        for source, m in per_source.items():
            if datasets is not None and source not in datasets:
                continue
            row: dict = {"model": model, "dataset": source, "seed": seed}
            for metric in metrics:
                v = m.get(metric)
                row[metric] = float(v) if v is not None else np.nan
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["model", "dataset", "seed"] + metrics)

    df = pd.DataFrame(rows, columns=["model", "dataset", "seed"] + metrics)
    df = df.sort_values(["model", "dataset", "seed"]).reset_index(drop=True)
    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-dir", default="outputs",
                   help="Root dir containing per-run subdirs (default: outputs)")
    p.add_argument("--dataset-tag", default="aggregate_binary",
                   help="Training-time tag in <model>__<tag>__seed<N>/ "
                        "(default: aggregate_binary)")
    p.add_argument("--datasets", default=None,
                   help="Comma-separated per-source datasets to KEEP in the output, "
                        "e.g. 'chemprot,drugprot,ddi,bc5cdr'. Filters the per_source "
                        "keys inside each eval JSON. Default: keep all sources present.")
    p.add_argument("--split", default="test", help="Which JSON split to pull (default: test)")
    p.add_argument("--view", choices=["pivot", "raw"], default="pivot",
                   help="'pivot' (default): model x dataset table averaged across seeds. "
                        "'raw': one row per (model, dataset, seed) with one column per metric.")
    p.add_argument("--metric", default="f1_positive", choices=_AVAILABLE_METRICS,
                   help="[pivot mode] Which per-source metric to place in the cells (default: f1_positive)")
    p.add_argument("--metrics", default="micro_f1,macro_f1,accuracy",
                   help="[raw mode] Comma-separated list of metric columns to include "
                        "(default: micro_f1,macro_f1,accuracy)")
    p.add_argument("--include-std", action="store_true",
                   help="[pivot mode] Also emit a std table and n_seeds; formats as mean ± std when writing")
    p.add_argument("--output-file", default=None,
                   help="Write to CSV / Markdown / JSON here; extension chooses the format. If omitted, prints the mean table to stdout.")
    p.add_argument("--precision", type=int, default=4, help="Decimal precision for formatting")
    args = p.parse_args()

    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    if not outputs_dir.is_dir():
        print(f"error: {outputs_dir} is not a directory", file=sys.stderr)
        return 2

    datasets_filter: set[str] | None = None
    if args.datasets:
        datasets_filter = {d.strip() for d in args.datasets.split(",") if d.strip()}

    # ── Raw view: one row per (model, dataset, seed), no aggregation ────────
    if args.view == "raw":
        metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
        unknown = [m for m in metric_list if m not in _AVAILABLE_METRICS]
        if unknown:
            print(
                f"error: unknown metric(s) in --metrics: {unknown}. "
                f"Choose from {_AVAILABLE_METRICS}",
                file=sys.stderr,
            )
            return 2

        raw_df = collect_raw(
            outputs_dir=outputs_dir,
            dataset_tag=args.dataset_tag,
            split=args.split,
            metrics=metric_list,
            datasets=datasets_filter,
        )
        if raw_df.empty:
            print(
                f"no aggregate_eval_{args.split}.json files found under {outputs_dir} "
                f"for dataset_tag={args.dataset_tag!r}",
                file=sys.stderr,
            )
            return 2

        # Round metric columns for display; leave (model, dataset, seed) untouched
        display_df = raw_df.copy()
        for m in metric_list:
            display_df[m] = display_df[m].round(args.precision)

        out = args.output_file
        if out is None:
            try:
                print(display_df.to_string(index=False))
            except Exception:
                print(display_df)
            return 0

        out_path = Path(out).expanduser().resolve()
        ext = out_path.suffix.lower()
        if ext == ".csv":
            display_df.to_csv(out_path, index=False)
        elif ext in (".md", ".markdown"):
            out_path.write_text(display_df.to_markdown(index=False) + "\n")
        elif ext == ".json":
            payload = {
                "view": "raw",
                "dataset_tag": args.dataset_tag,
                "split": args.split,
                "metrics": metric_list,
                "rows": raw_df.replace({np.nan: None}).to_dict(orient="records"),
            }
            out_path.write_text(json.dumps(payload, indent=2, default=str))
        elif ext in (".xlsx", ".xls"):
            with pd.ExcelWriter(out_path) as w:
                raw_df.to_excel(w, sheet_name="raw", index=False)
        else:
            out_path.write_text(display_df.to_string(index=False) + "\n")
        print(f"wrote {out_path}", file=sys.stderr)
        return 0

    # ── Pivot view (default, unchanged behavior) ────────────────────────────
    mean_df, std_df, n_df = collect(
        outputs_dir=outputs_dir,
        dataset_tag=args.dataset_tag,
        split=args.split,
        metric=args.metric,
        datasets=datasets_filter,
    )
    if mean_df.empty:
        print(
            f"no aggregate_eval_{args.split}.json files found under {outputs_dir} "
            f"for dataset_tag={args.dataset_tag!r}",
            file=sys.stderr,
        )
        return 2

    # If requested, produce a display table with mean ± std cells.
    if args.include_std:
        prec = args.precision
        display_df = mean_df.copy().astype(object)
        for model in mean_df.index:
            for source in mean_df.columns:
                m = mean_df.at[model, source]
                s = std_df.at[model, source]
                if pd.isna(m):
                    display_df.at[model, source] = "—"
                else:
                    display_df.at[model, source] = f"{m:.{prec}f} ± {s:.{prec}f}"
        # Attach n_seeds as a trailing column for context
        display_df["n_seeds"] = n_df.max(axis=1)
    else:
        display_df = mean_df.round(args.precision)

    out = args.output_file
    if out is None:
        # stdout: print a readable table
        try:
            print(display_df.to_string())
        except Exception:
            print(display_df)
        return 0

    out_path = Path(out).expanduser().resolve()
    ext = out_path.suffix.lower()
    if ext == ".csv":
        display_df.to_csv(out_path)
    elif ext in (".md", ".markdown"):
        out_path.write_text(display_df.to_markdown() + "\n")
    elif ext == ".json":
        payload = {
            "metric": args.metric,
            "dataset_tag": args.dataset_tag,
            "split": args.split,
            "mean": mean_df.replace({np.nan: None}).to_dict(orient="index"),
            "std": std_df.replace({np.nan: None}).to_dict(orient="index"),
            "n_seeds": n_df.astype("object").where(n_df.notna(), None).to_dict(orient="index"),
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
    elif ext in (".xlsx", ".xls"):
        with pd.ExcelWriter(out_path) as w:
            mean_df.to_excel(w, sheet_name="mean")
            std_df.to_excel(w, sheet_name="std")
            n_df.to_excel(w, sheet_name="n_seeds")
    else:
        # Fall back to a plain text dump
        out_path.write_text(display_df.to_string() + "\n")
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
