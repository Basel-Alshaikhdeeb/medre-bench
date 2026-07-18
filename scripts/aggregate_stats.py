#!/usr/bin/env python3
"""Introspect the aggregate dataset: per-source, per-class, per-split counts.

Loads each source dataset's split, applies the same entity-type remapping and
class assignment logic used by :class:`medre_bench.datasets.aggregate.AggregateDataset`,
and reports:

  * how many positives each source contributes AFTER remap (before sampling)
  * how many NO_RELATION examples each source can offer AFTER remap
  * the aggregate class distribution across all positives
  * the negative-sampling quotas that would apply (stratified 1:1 per source)
  * the final class distribution of the sampled train corpus

Results are cached under ``<outputs>/_aggregate_stats/<split>.json`` so subsequent
queries are instant. Pass ``--refresh`` to recompute from source.

Example:

    # All splits, tabular report to stdout
    python scripts/aggregate_stats.py

    # Just train, restricted to two sources, exported as CSV
    python scripts/aggregate_stats.py \\
        --split train --sources bc5cdr,biored \\
        --format csv --output-file agg_train_stats.csv

    # Just the final per-class counts of the validation split
    python scripts/aggregate_stats.py --split validation --view classes

    # JSON for programmatic use
    python scripts/aggregate_stats.py --split train --format json > train.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _build_stats(split: str, sources: list[str]) -> dict[str, Any]:
    """Compute per-source / per-class stats for one aggregate split.

    Reuses the private remap helpers from ``aggregate.py`` so entity-type mapping
    and class assignment stay in sync with the actual training data.
    """
    import medre_bench.datasets  # noqa: F401 - registration
    from medre_bench.datasets.aggregate import (
        AggregateDataset,
        _ENTITY_TYPE_MAP,
        _canonicalize_pair,
        _remap_negative,
        _remap_positive,
    )
    from medre_bench.registry import DATASET_REGISTRY

    per_source: dict[str, dict[str, Any]] = {}
    total_positives: list = []
    total_negatives: list = []

    for src in sources:
        try:
            ds_cls = DATASET_REGISTRY.get(src)
        except KeyError:
            per_source[src] = {"error": "not registered"}
            continue

        try:
            examples = ds_cls().load_split(split)
        except Exception as exc:  # noqa: BLE001
            per_source[src] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        raw_pos = raw_neg = 0
        pos_kept: list = []
        neg_kept: list = []
        # Per-source breakdowns of what got dropped and why
        pos_by_class: Counter[str] = Counter()
        neg_by_pair: Counter[str] = Counter()
        dropped_pos_type_pairs: Counter[str] = Counter()
        dropped_neg_type_pairs: Counter[str] = Counter()

        for ex in examples:
            if ex.label_id == 0:
                raw_neg += 1
                remapped = _remap_negative(ex)
                if remapped is None:
                    ct1 = _ENTITY_TYPE_MAP.get(ex.entity1_type, f"OOS({ex.entity1_type})")
                    ct2 = _ENTITY_TYPE_MAP.get(ex.entity2_type, f"OOS({ex.entity2_type})")
                    dropped_neg_type_pairs[f"{ct1}-{ct2}"] += 1
                    continue
                neg_kept.append(remapped)
                neg_by_pair[f"{remapped.entity1_type}-{remapped.entity2_type}"] += 1
            else:
                raw_pos += 1
                remapped = _remap_positive(ex)
                if remapped is None:
                    ct1 = _ENTITY_TYPE_MAP.get(ex.entity1_type, f"OOS({ex.entity1_type})")
                    ct2 = _ENTITY_TYPE_MAP.get(ex.entity2_type, f"OOS({ex.entity2_type})")
                    dropped_pos_type_pairs[f"{ct1}-{ct2}"] += 1
                    continue
                pos_kept.append(remapped)
                pos_by_class[remapped.label] += 1

        per_source[src] = {
            "raw_pos": raw_pos,
            "raw_neg": raw_neg,
            "pos_kept": len(pos_kept),
            "neg_available": len(neg_kept),
            "pos_dropped": raw_pos - len(pos_kept),
            "neg_dropped": raw_neg - len(neg_kept),
            "positives_by_class": dict(pos_by_class),
            "negatives_by_type_pair": dict(neg_by_pair),
            "dropped_positive_type_pairs": dict(dropped_pos_type_pairs.most_common()),
            "dropped_negative_type_pairs": dict(dropped_neg_type_pairs.most_common()),
        }
        total_positives.extend(pos_kept)
        total_negatives.extend(neg_kept)

    # Stratified negative sampling quotas (mirror aggregate.py exactly)
    neg_quotas: dict[str, int] = {}
    neg_available_by_source: dict[str, int] = {}
    for src, stats in per_source.items():
        if "error" in stats:
            continue
        neg_available_by_source[src] = stats["neg_available"]
        neg_quotas[src] = min(stats["pos_kept"], stats["neg_available"])

    target_total_neg = sum(s["pos_kept"] for s in per_source.values() if "error" not in s)
    shortfall = target_total_neg - sum(neg_quotas.values())
    if shortfall > 0:
        headroom = {
            s: neg_available_by_source[s] - neg_quotas[s]
            for s in neg_quotas
            if neg_available_by_source[s] - neg_quotas[s] > 0
        }
        total_h = sum(headroom.values())
        if total_h > 0:
            items = sorted(headroom.items())
            allocated = 0
            for i, (s, h) in enumerate(items):
                is_last = i == len(items) - 1
                take = (shortfall - allocated) if is_last else int(shortfall * h / total_h)
                take = min(take, h)
                neg_quotas[s] += take
                allocated += take

    # Attach per-source sampled counts + aggregate summaries
    for src, stats in per_source.items():
        if "error" in stats:
            continue
        stats["neg_sampled_quota"] = int(neg_quotas.get(src, 0))

    class_distribution_final = Counter()
    for src, stats in per_source.items():
        if "error" in stats:
            continue
        for cls, n in stats["positives_by_class"].items():
            class_distribution_final[cls] += n
    total_pos = sum(class_distribution_final.values())
    class_distribution_final["NO_RELATION"] = int(sum(neg_quotas.values()))

    total_class_distribution_available = Counter()
    for src, stats in per_source.items():
        if "error" in stats:
            continue
        for cls, n in stats["positives_by_class"].items():
            total_class_distribution_available[cls] += n
    total_class_distribution_available["NO_RELATION"] = sum(
        s["neg_available"] for s in per_source.values() if "error" not in s
    )

    return {
        "split": split,
        "sources": sources,
        "aggregate_labels": AggregateDataset().label_names(),
        "per_source": per_source,
        "totals": {
            "positives_kept": total_pos,
            "negatives_available": sum(s["neg_available"] for s in per_source.values() if "error" not in s),
            "negatives_sampled": int(sum(neg_quotas.values())),
            "combined_final_size": total_pos + int(sum(neg_quotas.values())),
        },
        "class_distribution_available_all_negatives": dict(total_class_distribution_available.most_common()),
        "class_distribution_final_sampled": dict(class_distribution_final.most_common()),
    }


def _load_or_compute(split: str, sources: list[str], cache_dir: Path, refresh: bool) -> dict[str, Any]:
    key = "_".join(sorted(sources)) if sources else "default"
    cache_file = cache_dir / f"{split}__{key}.json"
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats = _build_stats(split, sources)
    cache_file.write_text(json.dumps(stats, indent=2))
    return stats


# ─── Formatters ────────────────────────────────────────────────────────────────

def _fmt_int(n: Any) -> str:
    if n is None:
        return "—"
    if isinstance(n, (int, float)):
        return f"{int(n):,}"
    return str(n)


def _render_table_overview(stats: dict[str, Any]) -> str:
    lines = [f"=== Aggregate split: {stats['split']} ===", ""]
    lines.append(
        f"Total positives kept (after remap): {_fmt_int(stats['totals']['positives_kept'])}"
    )
    lines.append(
        f"Total negatives available:          {_fmt_int(stats['totals']['negatives_available'])}"
    )
    lines.append(
        f"Negatives sampled (stratified):     {_fmt_int(stats['totals']['negatives_sampled'])}"
    )
    lines.append(
        f"Final corpus size:                  {_fmt_int(stats['totals']['combined_final_size'])}"
    )
    lines.append("")

    # Per-source table
    rows = [(
        "source",
        "raw pos",
        "kept pos",
        "raw neg",
        "kept neg (avail)",
        "neg sampled quota",
    )]
    for src, s in stats["per_source"].items():
        if "error" in s:
            rows.append((src, s["error"], "", "", "", ""))
            continue
        rows.append((
            src,
            _fmt_int(s["raw_pos"]),
            _fmt_int(s["pos_kept"]),
            _fmt_int(s["raw_neg"]),
            _fmt_int(s["neg_available"]),
            _fmt_int(s["neg_sampled_quota"]),
        ))
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for i, r in enumerate(rows):
        lines.append("  ".join(str(v).ljust(widths[c]) for c, v in enumerate(r)))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    lines.append("")

    # Class distributions
    lines.append("Positives by aggregate class (pre-sampling, all sources):")
    for cls, n in stats["class_distribution_available_all_negatives"].items():
        if cls == "NO_RELATION":
            continue
        pct = 100.0 * n / max(stats["totals"]["positives_kept"], 1)
        lines.append(f"  {cls:<32s} {_fmt_int(n):>12s}  ({pct:5.2f}%)")
    lines.append("")

    lines.append("Final sampled corpus class distribution (post 1:1 sampling):")
    total = stats["totals"]["combined_final_size"]
    for cls, n in stats["class_distribution_final_sampled"].items():
        pct = 100.0 * n / max(total, 1)
        lines.append(f"  {cls:<32s} {_fmt_int(n):>12s}  ({pct:5.2f}%)")
    return "\n".join(lines)


def _render_table_per_source_classes(stats: dict[str, Any]) -> str:
    """Table: rows = source, columns = aggregate class, cells = positive count."""
    labels = [c for c in stats["aggregate_labels"] if c != "NO_RELATION"]
    rows = [["source"] + labels + ["total"]]
    for src, s in stats["per_source"].items():
        if "error" in s:
            rows.append([src, s["error"]] + [""] * len(labels))
            continue
        pos = s["positives_by_class"]
        row_vals = [_fmt_int(pos.get(c, 0)) for c in labels]
        total = sum(pos.get(c, 0) for c in labels)
        rows.append([src] + row_vals + [_fmt_int(total)])

    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    out = []
    for i, r in enumerate(rows):
        out.append("  ".join(str(v).ljust(widths[c]) for c, v in enumerate(r)))
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def _render_classes_only(stats: dict[str, Any]) -> str:
    lines = [f"=== {stats['split']} — Final sampled class distribution ==="]
    for cls, n in stats["class_distribution_final_sampled"].items():
        pct = 100.0 * n / max(stats["totals"]["combined_final_size"], 1)
        lines.append(f"  {cls:<32s} {_fmt_int(n):>12s}  ({pct:5.2f}%)")
    return "\n".join(lines)


def _render_csv(stats: dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    labels = [c for c in stats["aggregate_labels"] if c != "NO_RELATION"]
    w.writerow(["split", "source", "raw_pos", "kept_pos", "raw_neg", "neg_available",
                "neg_sampled_quota"] + labels)
    for src, s in stats["per_source"].items():
        if "error" in s:
            w.writerow([stats["split"], src, s["error"]] + [""] * (5 + len(labels)))
            continue
        pos = s["positives_by_class"]
        w.writerow([
            stats["split"], src,
            s["raw_pos"], s["pos_kept"], s["raw_neg"], s["neg_available"],
            s["neg_sampled_quota"],
        ] + [pos.get(c, 0) for c in labels])
    return buf.getvalue().rstrip()


def _render_markdown(stats: dict[str, Any]) -> str:
    labels = [c for c in stats["aggregate_labels"] if c != "NO_RELATION"]
    header = "| source | raw pos | kept pos | raw neg | neg available | neg sampled | " + " | ".join(labels) + " |"
    sep = "|" + "|".join(["---"] * (7 + len(labels))) + "|"
    body = []
    for src, s in stats["per_source"].items():
        if "error" in s:
            body.append(f"| {src} | {s['error']} " + "|  " * (5 + len(labels)) + "|")
            continue
        pos = s["positives_by_class"]
        cells = [
            src,
            _fmt_int(s["raw_pos"]),
            _fmt_int(s["pos_kept"]),
            _fmt_int(s["raw_neg"]),
            _fmt_int(s["neg_available"]),
            _fmt_int(s["neg_sampled_quota"]),
        ] + [_fmt_int(pos.get(c, 0)) for c in labels]
        body.append("| " + " | ".join(cells) + " |")
    parts = [
        f"### Aggregate split: `{stats['split']}`",
        "",
        f"**Totals**  final size={_fmt_int(stats['totals']['combined_final_size'])} "
        f"({_fmt_int(stats['totals']['positives_kept'])} positive + "
        f"{_fmt_int(stats['totals']['negatives_sampled'])} negative)",
        "",
        header, sep, *body,
        "",
        "**Final class distribution**",
        "",
        "| class | count | % |",
        "|---|---|---|",
    ]
    total = stats["totals"]["combined_final_size"]
    for cls, n in stats["class_distribution_final_sampled"].items():
        pct = 100.0 * n / max(total, 1)
        parts.append(f"| {cls} | {_fmt_int(n)} | {pct:.2f}% |")
    return "\n".join(parts)


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--split",
        default="all",
        help="'train', 'validation', 'test', or 'all' (default: all)",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated subset of source datasets; default = aggregate's own list (7 sources, no GAD)",
    )
    parser.add_argument(
        "--view",
        choices=["overview", "per_source_classes", "classes"],
        default="overview",
        help="Which slice of stats to print (default: overview)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv", "json", "markdown"],
        default="table",
    )
    parser.add_argument("--output-file", default=None)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory to cache computed stats JSON (default: outputs/_aggregate_stats)",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore cache; recompute from source")
    args = parser.parse_args()

    # Resolve source list from AggregateDataset unless caller overrode
    from medre_bench.datasets.aggregate import AggregateDataset

    if args.sources:
        source_list = [s.strip() for s in args.sources.split(",") if s.strip()]
    else:
        source_list = list(AggregateDataset.SOURCE_DATASETS)

    # Where to cache
    if args.cache_dir:
        cache_dir = Path(args.cache_dir).expanduser().resolve()
    else:
        cache_dir = Path("outputs").resolve() / "_aggregate_stats"

    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]

    outputs: list[str] = []
    for split in splits:
        stats = _load_or_compute(split, source_list, cache_dir, args.refresh)
        if args.format == "json":
            outputs.append(json.dumps(stats, indent=2))
        elif args.format == "csv":
            outputs.append(_render_csv(stats))
        elif args.format == "markdown":
            outputs.append(_render_markdown(stats))
        else:  # table
            if args.view == "classes":
                outputs.append(_render_classes_only(stats))
            elif args.view == "per_source_classes":
                outputs.append(f"=== {split}: positives per class per source ===")
                outputs.append(_render_table_per_source_classes(stats))
            else:
                outputs.append(_render_table_overview(stats))
        outputs.append("")

    rendered = "\n".join(outputs).rstrip() + "\n"
    if args.output_file:
        Path(args.output_file).write_text(rendered)
        print(f"wrote {args.output_file}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
