#!/usr/bin/env python3
"""Convert a per-run metrics CSV to a LaTeX table (models x datasets).

Reads a long-format CSV with columns [model, dataset, seed, micro_f1,
macro_f1, ...], aggregates across seeds, and emits a LaTeX ``tabular`` with
models on rows, datasets on columns, and cells formatted as
``mean $\\pm$ std``.

Works on any of the five paradigm CSVs under ``results/``:
``multiOriginal.csv``, ``multiTomek.csv``, ``BinaryOriginal.csv``,
``BinaryTomek.csv``, ``aggregate.csv``.

Examples::

    # Micro-F1 table from the multiclass unbalanced CSV, to stdout
    python scripts/csv_to_latex_table.py results/multiOriginal.csv --metric micro

    # Macro-F1 as percentages, bold the best in each column, write to .tex
    python scripts/csv_to_latex_table.py results/BinaryTomek.csv \\
        --metric macro --percent --bold-max \\
        --output tables/binary_balanced_macro.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMN: dict[str, str] = {
    "micro": "micro_f1",
    "macro": "macro_f1",
}


def _escape_latex(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def build_table(
    df: pd.DataFrame,
    metric_col: str,
    as_percent: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (mean, std, n_seeds) pivoted as model x dataset DataFrames."""
    if metric_col not in df.columns:
        raise ValueError(
            f"Column {metric_col!r} not in CSV. Available numeric columns: "
            f"{[c for c in df.columns if c not in ('model', 'dataset', 'seed')]}"
        )

    df = df.dropna(subset=[metric_col]).copy()
    scale = 100.0 if as_percent else 1.0

    grouped = df.groupby(["model", "dataset"])[metric_col]
    mean = grouped.mean().mul(scale).unstack("dataset")
    std = grouped.std(ddof=1).mul(scale).unstack("dataset").fillna(0.0)
    n = grouped.size().unstack("dataset").fillna(0).astype(int)
    return mean, std, n


def format_cells(
    mean: pd.DataFrame,
    std: pd.DataFrame,
    precision: int,
    bold_max: bool,
) -> pd.DataFrame:
    """Format each (mean, std) pair as ``mean $\\pm$ std``.

    When ``bold_max`` is set, the maximum mean within each dataset column
    is wrapped in ``\\textbf{...}``.
    """
    out = mean.copy().astype(object)
    for ds in mean.columns:
        col_max = mean[ds].max() if bold_max else None
        for model in mean.index:
            m = mean.at[model, ds]
            s = std.at[model, ds]
            if pd.isna(m):
                out.at[model, ds] = "---"
                continue
            cell = f"{m:.{precision}f} $\\pm$ {s:.{precision}f}"
            if bold_max and not pd.isna(col_max) and np.isclose(m, col_max):
                cell = f"\\textbf{{{cell}}}"
            out.at[model, ds] = cell
    return out


def df_to_latex(
    table: pd.DataFrame,
    caption: str,
    label: str,
    use_table_star: bool,
) -> str:
    cols = list(table.columns)
    col_spec = "l" + "c" * len(cols)
    env = "table*" if use_table_star else "table"

    header_cells = ["\\textbf{Model}"] + [f"\\textbf{{{_escape_latex(c)}}}" for c in cols]
    header = " & ".join(header_cells) + r" \\"

    lines = [
        f"\\begin{{{env}}}[t]",
        r"\centering",
        r"\footnotesize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for model in table.index:
        row_cells = [_escape_latex(model)] + [str(table.at[model, c]) for c in cols]
        lines.append(" & ".join(row_cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\end{{{env}}}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_csv", type=Path,
                   help="Long-format CSV (columns: model, dataset, seed, micro_f1, macro_f1, ...)")
    p.add_argument("--metric", choices=list(METRIC_COLUMN.keys()), default="micro",
                   help="Which F1 to tabulate (default: micro)")
    p.add_argument("--output", type=Path, default=None,
                   help="Write LaTeX here; prints to stdout if omitted")
    p.add_argument("--precision", type=int, default=3,
                   help="Decimal precision for both mean and std (default: 3)")
    p.add_argument("--percent", action="store_true",
                   help="Report values as percentages (multiply by 100)")
    p.add_argument("--bold-max", action="store_true",
                   help="Bold the highest mean in each dataset column")
    p.add_argument("--caption", default=None,
                   help="Table caption; auto-generated from CSV stem if omitted")
    p.add_argument("--label", default=None,
                   help="Table label; auto-generated from CSV stem if omitted")
    p.add_argument("--no-table-star", action="store_true",
                   help="Use 'table' environment instead of 'table*' (single-column papers)")
    p.add_argument("--sort-models", default=None,
                   help="Comma-separated model order; models not listed are appended alphabetically")
    args = p.parse_args()

    if not args.input_csv.exists():
        print(f"error: {args.input_csv} not found", file=sys.stderr)
        return 2

    df = pd.read_csv(args.input_csv)
    for required in ("model", "dataset", "seed"):
        if required not in df.columns:
            print(f"error: CSV missing required column {required!r}", file=sys.stderr)
            return 2

    metric_col = METRIC_COLUMN[args.metric]
    try:
        mean, std, n = build_table(df, metric_col, as_percent=args.percent)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.sort_models:
        preferred = [m.strip() for m in args.sort_models.split(",") if m.strip()]
        remaining = sorted(m for m in mean.index if m not in preferred)
        order = [m for m in preferred if m in mean.index] + remaining
        mean = mean.loc[order]
        std = std.loc[order]
        n = n.loc[order]

    formatted = format_cells(mean, std, args.precision, args.bold_max)

    stem = args.input_csv.stem
    pct_note = r" (\%)" if args.percent else ""
    caption = args.caption or (
        f"{args.metric.capitalize()}-F1{pct_note} mean $\\pm$ std over seeds "
        f"for the {_escape_latex(stem)} paradigm. "
        f"Seeds per cell: min={int(n.values.min())}, max={int(n.values.max())}."
    )
    label = args.label or f"tab:{stem}_{args.metric}"

    tex = df_to_latex(
        formatted,
        caption=caption,
        label=label,
        use_table_star=not args.no_table_star,
    )

    if args.output is None:
        print(tex)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tex)
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
