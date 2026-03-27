#!/usr/bin/env python3
"""Export results to various formats."""

import argparse
from pathlib import Path

from medre_bench.evaluation.analysis import compare_results


def main():
    parser = argparse.ArgumentParser(description="Export benchmark results")
    parser.add_argument("--results-dir", default="outputs", help="Results directory")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["table", "csv", "latex"],
        choices=["table", "csv", "latex"],
        help="Output formats",
    )
    parser.add_argument("--output-dir", default="exports", help="Export output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {"table": "txt", "csv": "csv", "latex": "tex"}

    for fmt in args.formats:
        output_file = output_dir / f"results.{extensions[fmt]}"
        print(f"\nExporting {fmt} to {output_file}...")
        compare_results(
            results_dir=args.results_dir,
            output_format=fmt,
            output_file=str(output_file),
        )


if __name__ == "__main__":
    main()
