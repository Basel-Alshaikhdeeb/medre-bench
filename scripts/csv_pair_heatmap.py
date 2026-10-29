#!/usr/bin/env python3
"""Render two side-by-side heatmaps from two per-run metrics CSVs.

* Panel 1 — absolute metric values for ``baseline_csv`` (mean across seeds).
* Panel 2 — ``comparison_csv`` as a *delta* against the baseline, i.e.
  ``mean(comparison) - mean(baseline)``. Cells carry ``+`` / ``-`` signs so
  the direction is readable without touching the colorbar.

Rows are models, columns are datasets, cells are mean-of-seeds for the
requested metric (``--metric micro`` or ``--metric macro``). The row and
column orders are shared across the two panels so the geometry lines up.

Any of the five paradigm CSVs under ``results/`` can be either argument.

Examples::

    # multiclass_balanced vs multiclass_unbalanced baseline (micro-F1)
    python scripts/csv_pair_heatmap.py \\
        results/multiOriginal.csv results/multiTomek.csv \\
        --metric micro --output results/micro_f1/multi_balanced_delta.png

    # same, macro, with custom labels
    python scripts/csv_pair_heatmap.py \\
        results/BinaryOriginal.csv results/BinaryTomek.csv \\
        --metric macro --output binary_delta.png \\
        --baseline-title "Binary / Unbalanced" --comparison-title "Binary / Balanced"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


METRIC_COLUMN: dict[str, str] = {
    "micro": "micro_f1",
    "macro": "macro_f1",
}


# Sequential sky-blue colormap used by both panels. Defined inline in the
# generated .tex so no extra packages (colorbrewer, etc.) are required.
RDBU_R_COLORMAP = r"""colormap={RdBuDiv}{
        rgb255=(240, 248, 252) rgb255=(210, 236, 250) rgb255=(190, 227, 247)
        rgb255=(169, 218, 245) rgb255=(151, 213, 247) rgb255=(128, 202, 242)
        rgb255=(104, 194, 242) rgb255=(88, 190, 245) rgb255=(64, 181, 245)
        rgb255=(44, 172, 242) rgb255=(22, 168, 247)
    }"""


def _escape_latex(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def _load_pivot(csv_path: Path, metric_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("model", "dataset", "seed", metric_col):
        if col not in df.columns:
            raise ValueError(f"{csv_path.name}: missing column {col!r}")
    df = df.dropna(subset=[metric_col])
    pivot = (
        df.groupby(["model", "dataset"])[metric_col]
          .mean()
          .unstack("dataset")
    )
    return pivot


def _align_axes(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    model_order: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (baseline, comparison) reindexed to the same (models, datasets)."""
    models_union = sorted(set(baseline.index) | set(comparison.index))
    if model_order:
        front = [m for m in model_order if m in models_union]
        rest = [m for m in models_union if m not in front]
        models_union = front + rest
    datasets_union = sorted(set(baseline.columns) | set(comparison.columns))

    return (
        baseline.reindex(index=models_union, columns=datasets_union),
        comparison.reindex(index=models_union, columns=datasets_union),
    )


def _annotate(ax, matrix: np.ndarray, fmt: str, text_color_fn) -> None:
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="black", fontsize=8)
                continue
            ax.text(j, i, format(v, fmt),
                    ha="center", va="center",
                    color=text_color_fn(v), fontsize=8)


def plot_pair(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    metric_label: str,
    baseline_title: str,
    comparison_title: str,
    output_path: Path,
    precision: int,
    as_percent: bool,
    baseline_cmap: str,
    delta_cmap: str,
) -> None:
    scale = 100.0 if as_percent else 1.0
    bl = baseline.values * scale
    delta = (comparison.values - baseline.values) * scale

    # Precision strings for annotations
    abs_fmt = f".{precision}f"
    delta_fmt = f"+.{precision}f"   # forces explicit +/- sign

    # Figure sizing: scale with the grid
    n_rows, n_cols = bl.shape
    fig_w = max(10.0, 0.9 * n_cols * 2 + 2)
    fig_h = max(4.0, 0.45 * n_rows + 1)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), constrained_layout=True)

    # ── panel 1: absolute baseline ─────────────────────────────────────────
    ax = axes[0]
    bl_min = np.nanmin(bl) if not np.all(np.isnan(bl)) else 0.0
    bl_max = np.nanmax(bl) if not np.all(np.isnan(bl)) else 1.0
    im0 = ax.imshow(bl, cmap=baseline_cmap, aspect="auto",
                    vmin=bl_min, vmax=bl_max)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(baseline.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(baseline.index, fontsize=9)
    ax.set_title(f"{baseline_title}\n{metric_label}", fontsize=11)
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    # Text contrast: dark text on light cells, white on dark cells
    span = bl_max - bl_min if bl_max > bl_min else 1.0
    threshold = bl_min + 0.55 * span
    _annotate(ax, bl, abs_fmt, lambda v, t=threshold: "white" if v > t else "black")

    # ── panel 2: delta (comparison - baseline) ─────────────────────────────
    ax = axes[1]
    abs_delta_max = np.nanmax(np.abs(delta)) if not np.all(np.isnan(delta)) else 1.0
    # Avoid vmin == vmax if deltas are all zero
    abs_delta_max = max(abs_delta_max, 1e-6)
    norm = TwoSlopeNorm(vmin=-abs_delta_max, vcenter=0.0, vmax=abs_delta_max)
    im1 = ax.imshow(delta, cmap=delta_cmap, aspect="auto", norm=norm)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(baseline.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(baseline.index, fontsize=9)
    ax.set_title(f"{comparison_title} $-$ {baseline_title}\n$\\Delta$ {metric_label}",
                 fontsize=11)
    fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    # Text contrast on diverging palette: white when cell is strongly colored
    delta_threshold = 0.55 * abs_delta_max
    _annotate(ax, delta, delta_fmt,
              lambda v, t=delta_threshold: "white" if abs(v) > t else "black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"wrote {output_path}", file=sys.stderr)
    plt.close(fig)


_LATEX_TEMPLATE = r"""%% Preamble requirements:
%%   \usepackage{pgfplots}
%%   \pgfplotsset{compat=1.18}
%%   \usepgfplotslibrary{groupplots, colormaps}
%% Sequential sky-blue colormap "RdBuDiv" is defined inline below.
\begin{%(env)s}[t]
\centering
\begin{tikzpicture}
\pgfplotsset{
    %(rdbu_colormap)s,
    heatmap base/.style={
        colorbar,
        colorbar style={width=0.25cm, font=\scriptsize},
        enlargelimits=false,
        y dir=reverse,
        xtick={%(xticks)s},
        xticklabels={%(ds_labels)s},
        xticklabel style={rotate=30, anchor=east, font=\scriptsize},
        ytick={%(yticks)s},
        yticklabels={%(model_labels)s},
        yticklabel style={font=\scriptsize},
        title style={font=\small, align=center},
        every node near coord/.append style={anchor=center, inner sep=1pt, font=\tiny},
    },
}
\begin{groupplot}[
    group style={group size=2 by 1, horizontal sep=3.5cm},
    width=%(axis_width)s, height=%(axis_height)s,
]
\nextgroupplot[
    heatmap base,
    colormap name=RdBuDiv,
    title={%(baseline_title)s\\%(metric_label)s},
    point meta min=%(bl_min).6f, point meta max=%(bl_max).6f,
    nodes near coords={\pgfmathprintnumber[fixed, fixed zerofill, precision=%(precision)d]{\pgfplotspointmeta}},
]
\addplot[matrix plot*, mesh/cols=%(n_cols)d, point meta=explicit] table[meta=val] {
%(bl_table)s
};

\nextgroupplot[
    heatmap base,
    colormap name=RdBuDiv,
    title={%(comparison_title)s $-$ %(baseline_title)s\\$\Delta$ %(metric_label)s},
    point meta min=%(neg_delta_abs).6f, point meta max=%(delta_abs).6f,
    nodes near coords={\pgfmathprintnumber[fixed, fixed zerofill, precision=%(precision)d, print sign]{\pgfplotspointmeta}},
]
\addplot[matrix plot*, mesh/cols=%(n_cols)d, point meta=explicit] table[meta=val] {
%(delta_table)s
};
\end{groupplot}
\end{tikzpicture}
\caption{%(caption)s}
\label{%(label)s}
\end{%(env)s}
"""


def _matrix_data_block(values: np.ndarray) -> tuple[str, int]:
    """Serialize a 2D array as a pgfplots matrix-plot table.

    NaN cells are substituted with 0.0 so ``matrix plot*`` keeps a regular
    grid; the return value also reports the substitution count so the caller
    can warn.
    """
    rows = ["x y val"]
    nan_count = 0
    n_rows, n_cols = values.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if np.isnan(v):
                nan_count += 1
                rows.append(f"{j} {i} 0")
            else:
                rows.append(f"{j} {i} {v:.6f}")
    return "\n".join(rows), nan_count


def pair_heatmap_latex(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    metric_label: str,
    baseline_title: str,
    comparison_title: str,
    precision: int,
    as_percent: bool,
    caption: str,
    label: str,
    use_figure_star: bool,
) -> str:
    scale = 100.0 if as_percent else 1.0
    bl = baseline.values.astype(float) * scale
    delta = (comparison.values.astype(float) - baseline.values.astype(float)) * scale
    n_rows, n_cols = bl.shape

    bl_min = float(np.nanmin(bl)) if not np.all(np.isnan(bl)) else 0.0
    bl_max = float(np.nanmax(bl)) if not np.all(np.isnan(bl)) else 1.0
    if bl_min == bl_max:
        bl_max = bl_min + 1e-6

    delta_abs = float(np.nanmax(np.abs(delta))) if not np.all(np.isnan(delta)) else 1.0
    delta_abs = max(delta_abs, 1e-6)

    bl_table, bl_nan = _matrix_data_block(bl)
    delta_table, delta_nan = _matrix_data_block(delta)
    if bl_nan or delta_nan:
        print(
            f"warning: substituted 0.0 for {bl_nan + delta_nan} missing cells "
            "to keep matrix plot* grid regular (check your CSV coverage)",
            file=sys.stderr,
        )

    axis_width_cm = max(5.5, n_cols * 0.85)
    axis_height_cm = max(4.0, n_rows * 0.45)

    return _LATEX_TEMPLATE % {
        "env": "figure*" if use_figure_star else "figure",
        "rdbu_colormap": RDBU_R_COLORMAP,
        "xticks": ",".join(str(i) for i in range(n_cols)),
        "yticks": ",".join(str(i) for i in range(n_rows)),
        "ds_labels": ",".join(_escape_latex(d) for d in baseline.columns),
        "model_labels": ",".join(_escape_latex(m) for m in baseline.index),
        "baseline_title": _escape_latex(baseline_title),
        "comparison_title": _escape_latex(comparison_title),
        "metric_label": metric_label.replace("%", r"\%"),
        "precision": precision,
        "n_cols": n_cols,
        "bl_min": bl_min,
        "bl_max": bl_max,
        "neg_delta_abs": -delta_abs,
        "delta_abs": delta_abs,
        "bl_table": bl_table,
        "delta_table": delta_table,
        "axis_width": f"{axis_width_cm:.1f}cm",
        "axis_height": f"{axis_height_cm:.1f}cm",
        "caption": caption,
        "label": label,
    }


_LATEX_TEMPLATE_TWO_METRIC = r"""%% Preamble requirements:
%%   \usepackage{pgfplots}
%%   \pgfplotsset{compat=1.18}
%%   \usepgfplotslibrary{groupplots, colormaps}
%%   \usepackage{pdflscape}  %% only when --landscape is used
%% Sequential sky-blue colormap "RdBuDiv" is defined inline below.
%(opener)s
\centering
\begin{tikzpicture}
\pgfplotsset{
    %(rdbu_colormap)s,
    heatmap base/.style={
        colorbar,
        colorbar style={width=0.25cm, font=\scriptsize},
        enlargelimits=false,
        y dir=reverse,
        xtick={%(xticks)s},
        xticklabels={%(ds_labels)s},
        xticklabel style={rotate=30, anchor=east, font=\scriptsize},
        ytick={%(yticks)s},
        yticklabels={%(model_labels)s},
        yticklabel style={font=\scriptsize},
        title style={font=\small, align=center},
        every node near coord/.append style={anchor=center, inner sep=1pt, font=\tiny},
    },
}
\begin{groupplot}[
    group style={
        group size=2 by 2,
        horizontal sep=4.0cm,
        vertical sep=2.0cm,
    },
    width=%(axis_width)s, height=%(axis_height)s,
]
%% ── row 1: micro ─────────────────────────────────────────
\nextgroupplot[
    heatmap base,
    colormap name=RdBuDiv,
    title={%(baseline_title)s\\micro-F1},
    xticklabels={},
    point meta min=%(bl_micro_min).6f, point meta max=%(bl_micro_max).6f,
    nodes near coords={\pgfmathprintnumber[fixed, fixed zerofill, precision=%(precision)d]{\pgfplotspointmeta}},
]
\addplot[matrix plot*, mesh/cols=%(n_cols)d, point meta=explicit] table[meta=val] {
%(bl_micro_table)s
};

\nextgroupplot[
    heatmap base,
    colormap name=RdBuDiv,
    title={%(comparison_title)s $-$ %(baseline_title)s\\$\Delta$ micro-F1},
    xticklabels={}, yticklabels={},
    point meta min=%(neg_delta_micro_abs).6f, point meta max=%(delta_micro_abs).6f,
    nodes near coords={\pgfmathprintnumber[fixed, fixed zerofill, precision=%(precision)d, print sign]{\pgfplotspointmeta}},
]
\addplot[matrix plot*, mesh/cols=%(n_cols)d, point meta=explicit] table[meta=val] {
%(delta_micro_table)s
};

%% ── row 2: macro ─────────────────────────────────────────
\nextgroupplot[
    heatmap base,
    colormap name=RdBuDiv,
    title={%(baseline_title)s\\macro-F1},
    point meta min=%(bl_macro_min).6f, point meta max=%(bl_macro_max).6f,
    nodes near coords={\pgfmathprintnumber[fixed, fixed zerofill, precision=%(precision)d]{\pgfplotspointmeta}},
]
\addplot[matrix plot*, mesh/cols=%(n_cols)d, point meta=explicit] table[meta=val] {
%(bl_macro_table)s
};

\nextgroupplot[
    heatmap base,
    colormap name=RdBuDiv,
    title={%(comparison_title)s $-$ %(baseline_title)s\\$\Delta$ macro-F1},
    yticklabels={},
    point meta min=%(neg_delta_macro_abs).6f, point meta max=%(delta_macro_abs).6f,
    nodes near coords={\pgfmathprintnumber[fixed, fixed zerofill, precision=%(precision)d, print sign]{\pgfplotspointmeta}},
]
\addplot[matrix plot*, mesh/cols=%(n_cols)d, point meta=explicit] table[meta=val] {
%(delta_macro_table)s
};
\end{groupplot}
\end{tikzpicture}
\caption{%(caption)s}
\label{%(label)s}
%(closer)s
"""


def _align_four(frames: list[pd.DataFrame], model_order: list[str] | None) -> list[pd.DataFrame]:
    """Reindex four pivots to a shared (models, datasets) grid."""
    models_union = sorted(set().union(*(f.index for f in frames)))
    if model_order:
        front = [m for m in model_order if m in models_union]
        rest = [m for m in models_union if m not in front]
        models_union = front + rest
    datasets_union = sorted(set().union(*(f.columns for f in frames)))
    return [f.reindex(index=models_union, columns=datasets_union) for f in frames]


def pair_heatmap_latex_two_metric(
    bl_micro: pd.DataFrame,
    cmp_micro: pd.DataFrame,
    bl_macro: pd.DataFrame,
    cmp_macro: pd.DataFrame,
    baseline_title: str,
    comparison_title: str,
    precision: int,
    as_percent: bool,
    caption: str,
    label: str,
    use_figure_star: bool,
    landscape: bool,
) -> str:
    scale = 100.0 if as_percent else 1.0
    bl_mi = bl_micro.values.astype(float) * scale
    dl_mi = (cmp_micro.values.astype(float) - bl_micro.values.astype(float)) * scale
    bl_ma = bl_macro.values.astype(float) * scale
    dl_ma = (cmp_macro.values.astype(float) - bl_macro.values.astype(float)) * scale

    n_rows, n_cols = bl_mi.shape

    def _range(arr: np.ndarray) -> tuple[float, float]:
        if np.all(np.isnan(arr)):
            return 0.0, 1.0
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if lo == hi:
            hi = lo + 1e-6
        return lo, hi

    def _abs_max(arr: np.ndarray) -> float:
        v = float(np.nanmax(np.abs(arr))) if not np.all(np.isnan(arr)) else 1.0
        return max(v, 1e-6)

    bl_mi_min, bl_mi_max = _range(bl_mi)
    bl_ma_min, bl_ma_max = _range(bl_ma)
    dl_mi_abs = _abs_max(dl_mi)
    dl_ma_abs = _abs_max(dl_ma)

    bl_mi_tbl, n1 = _matrix_data_block(bl_mi)
    dl_mi_tbl, n2 = _matrix_data_block(dl_mi)
    bl_ma_tbl, n3 = _matrix_data_block(bl_ma)
    dl_ma_tbl, n4 = _matrix_data_block(dl_ma)
    total_nan = n1 + n2 + n3 + n4
    if total_nan:
        print(
            f"warning: substituted 0.0 for {total_nan} missing cells "
            "to keep matrix plot* grids regular (check CSV coverage)",
            file=sys.stderr,
        )

    axis_width_cm = max(5.5, n_cols * 0.85)
    axis_height_cm = max(4.0, n_rows * 0.45)

    env_name = "figure*" if use_figure_star else "figure"
    if landscape:
        opener = f"\\begin{{landscape}}\n\\begin{{{env_name}}}[p]"
        closer = f"\\end{{{env_name}}}\n\\end{{landscape}}"
    else:
        opener = f"\\begin{{{env_name}}}[p]"
        closer = f"\\end{{{env_name}}}"

    return _LATEX_TEMPLATE_TWO_METRIC % {
        "opener": opener,
        "closer": closer,
        "rdbu_colormap": RDBU_R_COLORMAP,
        "xticks": ",".join(str(i) for i in range(n_cols)),
        "yticks": ",".join(str(i) for i in range(n_rows)),
        "ds_labels": ",".join(_escape_latex(d) for d in bl_micro.columns),
        "model_labels": ",".join(_escape_latex(m) for m in bl_micro.index),
        "baseline_title": _escape_latex(baseline_title),
        "comparison_title": _escape_latex(comparison_title),
        "precision": precision,
        "n_cols": n_cols,
        "bl_micro_min": bl_mi_min, "bl_micro_max": bl_mi_max,
        "bl_macro_min": bl_ma_min, "bl_macro_max": bl_ma_max,
        "neg_delta_micro_abs": -dl_mi_abs, "delta_micro_abs": dl_mi_abs,
        "neg_delta_macro_abs": -dl_ma_abs, "delta_macro_abs": dl_ma_abs,
        "bl_micro_table": bl_mi_tbl,
        "delta_micro_table": dl_mi_tbl,
        "bl_macro_table": bl_ma_tbl,
        "delta_macro_table": dl_ma_tbl,
        "axis_width": f"{axis_width_cm:.1f}cm",
        "axis_height": f"{axis_height_cm:.1f}cm",
        "caption": caption,
        "label": label,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("baseline_csv", type=Path,
                   help="Baseline CSV — rendered as absolute values in panel 1")
    p.add_argument("comparison_csv", type=Path,
                   help="Comparison CSV — rendered as (comparison - baseline) in panel 2")
    p.add_argument("--metric", choices=list(METRIC_COLUMN.keys()), default="micro",
                   help="Which F1 to tabulate (default: micro)")
    p.add_argument("--output", type=Path, default=Path("pair_heatmap.png"),
                   help="Where to save the figure (PNG/PDF/SVG by extension)")
    p.add_argument("--precision", type=int, default=3,
                   help="Decimal places for cell annotations (default: 3)")
    p.add_argument("--percent", action="store_true",
                   help="Multiply values by 100 so cells read as percentages")
    p.add_argument("--baseline-title", default=None,
                   help="Panel-1 title; defaults to baseline CSV filename")
    p.add_argument("--comparison-title", default=None,
                   help="Panel-2 title; defaults to comparison CSV filename")
    p.add_argument("--baseline-cmap", default="viridis",
                   help="Colormap for absolute values (default: viridis)")
    p.add_argument("--delta-cmap", default="RdBu_r",
                   help="Diverging colormap for deltas (default: RdBu_r)")
    p.add_argument("--sort-models", default=None,
                   help="Comma-separated model order; unlisted models go after, alphabetically")
    p.add_argument("--caption", default=None,
                   help="Caption for LaTeX output (.tex); auto-generated if omitted")
    p.add_argument("--label", default=None,
                   help="Label for LaTeX output (.tex); auto-generated if omitted")
    p.add_argument("--no-figure-star", action="store_true",
                   help="LaTeX output: use 'figure' instead of 'figure*' (single-column papers)")
    p.add_argument("--two-metric", action="store_true",
                   help="LaTeX output: emit a 2x2 figure with micro-F1 on the top row "
                        "and macro-F1 on the bottom; --metric is ignored in this mode")
    p.add_argument("--landscape", action="store_true",
                   help="LaTeX --two-metric only: wrap in \\begin{landscape}...\\end{landscape} "
                        "(requires \\usepackage{pdflscape})")
    args = p.parse_args()

    for path in (args.baseline_csv, args.comparison_csv):
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 2

    if args.two_metric:
        if args.output.suffix.lower() != ".tex":
            print("error: --two-metric requires --output ending in .tex",
                  file=sys.stderr)
            return 2
        model_order = (
            [m.strip() for m in args.sort_models.split(",") if m.strip()]
            if args.sort_models else None
        )
        baseline_title = args.baseline_title or args.baseline_csv.stem
        comparison_title = args.comparison_title or args.comparison_csv.stem
        try:
            bl_micro = _load_pivot(args.baseline_csv, "micro_f1")
            cmp_micro = _load_pivot(args.comparison_csv, "micro_f1")
            bl_macro = _load_pivot(args.baseline_csv, "macro_f1")
            cmp_macro = _load_pivot(args.comparison_csv, "macro_f1")
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        bl_micro, cmp_micro, bl_macro, cmp_macro = _align_four(
            [bl_micro, cmp_micro, bl_macro, cmp_macro], model_order,
        )
        stem = f"{args.baseline_csv.stem}_vs_{args.comparison_csv.stem}"
        caption = args.caption or (
            "Micro- and macro-F1"
            + (r" (\%)" if args.percent else "")
            + f" for {_escape_latex(baseline_title)} (left column) and the "
            + f"{_escape_latex(comparison_title)} $-$ {_escape_latex(baseline_title)} "
            + "delta (right column); micro on the top row, macro on the bottom."
        )
        label = args.label or f"fig:{stem}_micro_macro"
        tex = pair_heatmap_latex_two_metric(
            bl_micro=bl_micro,
            cmp_micro=cmp_micro,
            bl_macro=bl_macro,
            cmp_macro=cmp_macro,
            baseline_title=baseline_title,
            comparison_title=comparison_title,
            precision=args.precision,
            as_percent=args.percent,
            caption=caption,
            label=label,
            use_figure_star=not args.no_figure_star,
            landscape=args.landscape,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tex)
        print(f"wrote {args.output}", file=sys.stderr)
        return 0

    metric_col = METRIC_COLUMN[args.metric]
    try:
        baseline = _load_pivot(args.baseline_csv, metric_col)
        comparison = _load_pivot(args.comparison_csv, metric_col)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    model_order = (
        [m.strip() for m in args.sort_models.split(",") if m.strip()]
        if args.sort_models else None
    )
    baseline, comparison = _align_axes(baseline, comparison, model_order)

    if baseline.shape != comparison.shape or not (
        all(baseline.index == comparison.index)
        and all(baseline.columns == comparison.columns)
    ):
        print("error: baseline and comparison grids could not be aligned", file=sys.stderr)
        return 2

    metric_label = f"{args.metric}-F1" + (" (%)" if args.percent else "")
    baseline_title = args.baseline_title or args.baseline_csv.stem
    comparison_title = args.comparison_title or args.comparison_csv.stem

    if args.output.suffix.lower() == ".tex":
        stem = f"{args.baseline_csv.stem}_vs_{args.comparison_csv.stem}"
        caption = args.caption or (
            f"{args.metric.capitalize()}-F1"
            + (r" (\%)" if args.percent else "")
            + f" for {_escape_latex(baseline_title)} (left) and the "
            + f"{_escape_latex(comparison_title)} $-$ {_escape_latex(baseline_title)} "
            + "delta (right)."
        )
        label = args.label or f"fig:{stem}_{args.metric}"
        tex = pair_heatmap_latex(
            baseline=baseline,
            comparison=comparison,
            metric_label=metric_label,
            baseline_title=baseline_title,
            comparison_title=comparison_title,
            precision=args.precision,
            as_percent=args.percent,
            caption=caption,
            label=label,
            use_figure_star=not args.no_figure_star,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(tex)
        print(f"wrote {args.output}", file=sys.stderr)
        return 0

    plot_pair(
        baseline=baseline,
        comparison=comparison,
        metric_label=metric_label,
        baseline_title=baseline_title,
        comparison_title=comparison_title,
        output_path=args.output,
        precision=args.precision,
        as_percent=args.percent,
        baseline_cmap=args.baseline_cmap,
        delta_cmap=args.delta_cmap,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
