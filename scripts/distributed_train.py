"""Lightweight entry point for distributed training via torchrun."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--local_rank", type=int, default=0)  # torchrun injects this
    args, remaining = parser.parse_known_args()

    # Parse dot-notation overrides from remaining args
    overrides = {}
    for item in remaining:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        parts = key.split(".")
        d = overrides
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value

    overrides["output_dir"] = args.output_dir

    from medre_bench.config.loader import load_config
    from medre_bench.training.trainer import run_training

    cfg = load_config(
        model=args.model,
        dataset=args.dataset,
        config_path=args.config,
        overrides=overrides,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
