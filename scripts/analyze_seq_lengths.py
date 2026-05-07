#!/usr/bin/env python3
"""Profile token-length distribution per dataset to pick max_seq_length.

Applies the project's sentence-level + entity-marker preprocessing, then
tokenizes every example (across train/val/test by default) with one or more
reference tokenizers. Reports per-dataset percentiles plus a recommendation:
the longest observed length (rounded up to a multiple of 8) across all
profiled tokenizers, so a single value is safe for every model in the
registry.

Usage:
    python scripts/analyze_seq_lengths.py
    python scripts/analyze_seq_lengths.py --datasets chemprot,drugprot
    python scripts/analyze_seq_lengths.py --tokenizers bert-base-uncased
"""
from __future__ import annotations

import argparse

import numpy as np
from transformers import AutoTokenizer

import medre_bench.datasets  # noqa: F401  triggers dataset registration
from medre_bench.datasets.base import apply_entity_markers
from medre_bench.registry import DATASET_REGISTRY


def _marked_texts(examples, strategy: str) -> list[str]:
    out: list[str] = []
    for ex in examples:
        has_entities = (ex.entity1_start != ex.entity1_end) or (
            ex.entity2_start != ex.entity2_end
        )
        if has_entities and strategy != "none":
            t = apply_entity_markers(
                text=ex.text,
                e1_start=ex.entity1_start, e1_end=ex.entity1_end, e1_type=ex.entity1_type,
                e2_start=ex.entity2_start, e2_end=ex.entity2_end, e2_type=ex.entity2_type,
                strategy=strategy,
            )
        else:
            t = ex.text
        out.append(t)
    return out


def _round_up(x: int, multiple: int = 8) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset keys; default: all registered")
    parser.add_argument(
        "--tokenizers",
        default="bert-base-uncased,roberta-base",
        help="Comma-separated tokenizers to profile (worst case across them is the recommendation)",
    )
    parser.add_argument("--splits", default="train,validation,test",
                        help="Splits to include in the length population")
    parser.add_argument("--strategy", default="typed_entity_marker_punct",
                        help="Entity marker strategy (default matches training default)")
    args = parser.parse_args()

    if args.datasets:
        names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    else:
        names = sorted(DATASET_REGISTRY.list_available())

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    tok_names = [t.strip() for t in args.tokenizers.split(",") if t.strip()]
    print(f"Loading tokenizers: {tok_names}")
    tokenizers = {name: AutoTokenizer.from_pretrained(name) for name in tok_names}

    header = f"{'dataset':<12} {'tokenizer':<22} {'n':>8} {'p50':>5} {'p95':>5} {'p99':>5} {'max':>5}  rec→"
    print()
    print(header)
    print("-" * len(header))

    summary: dict[str, int] = {}
    for ds_name in names:
        try:
            ds_cls = DATASET_REGISTRY.get(ds_name)
            ds = ds_cls()
        except Exception as e:
            print(f"{ds_name:<12} ERROR loading dataset class: {e}")
            continue

        examples = []
        for split in splits:
            try:
                examples.extend(ds.load_split(split))
            except Exception as e:
                print(f"{ds_name:<12} skip split {split!r}: {type(e).__name__}: {e}")

        if not examples:
            print(f"{ds_name:<12} (no examples loaded)")
            continue

        texts = _marked_texts(examples, args.strategy)

        rec_per_tok = []
        for tname, tok in tokenizers.items():
            enc = tok(texts, add_special_tokens=True, truncation=False, padding=False)
            lens = np.array([len(ids) for ids in enc["input_ids"]])
            p50 = int(np.percentile(lens, 50))
            p95 = int(np.percentile(lens, 95))
            p99 = int(np.percentile(lens, 99))
            mx = int(lens.max())
            rec_per_tok.append(mx)
            print(f"{ds_name:<12} {tname:<22} {len(lens):>8} {p50:>5} {p95:>5} {p99:>5} {mx:>5}")

        summary[ds_name] = _round_up(max(rec_per_tok))
        print(f"{ds_name:<12} {'->>RECOMMEND (max, x8)':<22} {'':>8} {'':>5} {'':>5} {'':>5} {summary[ds_name]:>5}")
        print()

    print("=" * 60)
    print("Recommended max_seq_length per dataset (covers every profiled tokenizer):")
    for ds_name, rec in summary.items():
        print(f"  {ds_name:<12} {rec}")


if __name__ == "__main__":
    main()
