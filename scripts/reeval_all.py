#!/usr/bin/env python3
"""Walk an outputs tree and run ``medre-bench evaluate`` for every run.

For each run of the form::

    <outputs>/<run_name>/<timestamp>/checkpoints/best/

the script reads ``<timestamp>/config_snapshot.yaml`` to recover
``model``, ``dataset``, and ``seed``, then invokes::

    medre-bench evaluate --checkpoint <ckpt> --dataset <ds> \
        --split <split> --output-dir <ts>/<out_subdir>

so that ``metrics_<ds>_<split>.json`` (and ``predictions_<ds>_<split>.json``)
land under each run's own folder. Skips a run when the target metrics file
already exists, unless ``--overwrite`` is passed.

Examples::

    # Re-eval every multiOriginal/multiTomek checkpoint
    python scripts/reeval_all.py \\
        --outputs-dir outputs \\
        --pattern 'multiOriginal|multiTomek' \\
        --out-subdir binarized_eval

    # Dry-run first — print the commands only
    python scripts/reeval_all.py --dry-run --pattern multiOriginal

    # If `medre-bench` isn't on PATH, use the module entrypoint
    python scripts/reeval_all.py --cmd 'python -m medre_bench evaluate'
"""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _latest_ts_with_checkpoint(run_dir: Path) -> Path | None:
    """Newest timestamped subdir of ``run_dir`` that contains checkpoints/best/."""
    cands = [
        d for d in run_dir.iterdir()
        if d.is_dir() and (d / "checkpoints" / "best").is_dir()
    ]
    if not cands:
        return None
    return sorted(cands, key=lambda d: d.name)[-1]


def discover_runs(outputs_dir: Path, pattern: re.Pattern | None):
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        if pattern and not pattern.search(run_dir.name):
            continue
        ts = _latest_ts_with_checkpoint(run_dir)
        if ts is None:
            continue
        cfg_path = ts / "config_snapshot.yaml"
        if not cfg_path.exists():
            print(f"skip {run_dir.name}: no config_snapshot.yaml", file=sys.stderr)
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
        except Exception as exc:
            print(f"skip {run_dir.name}: cannot parse config: {exc}", file=sys.stderr)
            continue
        model = (cfg.get("model") or {}).get("name")
        dataset = (cfg.get("dataset") or {}).get("name")
        seed = (cfg.get("training") or {}).get("seed")
        if not (model and dataset):
            print(f"skip {run_dir.name}: missing model/dataset in config", file=sys.stderr)
            continue
        yield {
            "run_dir": run_dir,
            "ts_dir": ts,
            "ckpt": ts / "checkpoints" / "best",
            "model": model,
            "dataset": dataset,
            "seed": seed,
        }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--outputs-dir", type=Path, default=Path("outputs"),
                   help="Root dir containing per-run subdirs (default: outputs)")
    p.add_argument("--pattern", default=None,
                   help="Regex filter on run_dir basename "
                        "(e.g. 'multiOriginal|multiTomek'). Default: no filter.")
    p.add_argument("--split", default="test",
                   help="Dataset split to evaluate (default: test)")
    p.add_argument("--out-subdir", default="binarized_eval",
                   help="Subdir of each <timestamp>/ to write eval outputs into "
                        "(default: binarized_eval)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if metrics_<ds>_<split>.json already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands that would run, don't execute")
    p.add_argument("--cmd", default="medre-bench evaluate",
                   help="Base command prefix (default: 'medre-bench evaluate'). "
                        "Pass 'python -m medre_bench evaluate' if not on PATH.")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Append per-run status to this log file")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra args forwarded verbatim to each eval call "
                        "(put them after `--extra`)")
    args = p.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    if not outputs_dir.is_dir():
        print(f"error: {outputs_dir} is not a directory", file=sys.stderr)
        return 2

    pat = re.compile(args.pattern) if args.pattern else None
    runs = list(discover_runs(outputs_dir, pat))
    print(f"Discovered {len(runs)} runs under {outputs_dir}", file=sys.stderr)
    if not runs:
        return 0

    cmd_prefix = shlex.split(args.cmd)
    log = open(args.log_file, "a") if args.log_file else None

    n_ok = n_fail = n_skip = 0
    for i, r in enumerate(runs, 1):
        out_dir = r["ts_dir"] / args.out_subdir
        target = out_dir / f"metrics_{r['dataset']}_{args.split}.json"
        tag = f"{r['run_dir'].name}  (model={r['model']}, ds={r['dataset']}, seed={r['seed']})"

        if target.exists() and not args.overwrite:
            n_skip += 1
            print(f"[{i:3d}/{len(runs)}] skip  {tag} — {target.name} exists",
                  file=sys.stderr)
            if log:
                log.write(f"skip\t{r['run_dir'].name}\t{r['dataset']}\talready-done\n")
            continue

        cmd = [
            *cmd_prefix,
            "--checkpoint", str(r["ckpt"]),
            "--dataset", r["dataset"],
            "--split", args.split,
            "--output-dir", str(out_dir),
            *args.extra,
        ]
        print(f"[{i:3d}/{len(runs)}] run   {tag}", file=sys.stderr)
        if args.dry_run:
            print("    $ " + " ".join(shlex.quote(c) for c in cmd), file=sys.stderr)
            continue

        t0 = time.time()
        try:
            res = subprocess.run(cmd, check=False)
            ok = res.returncode == 0
        except Exception as exc:
            ok = False
            print(f"    ERROR launching: {exc}", file=sys.stderr)
        dt = time.time() - t0
        status = "ok" if ok else "fail"
        n_ok += ok
        n_fail += (not ok)
        print(f"    -> {status} ({dt:.1f}s)", file=sys.stderr)
        if log:
            log.write(f"{status}\t{r['run_dir'].name}\t{r['dataset']}\t{dt:.1f}s\n")

    if log:
        log.close()

    print(
        f"\nDone. {n_ok} ok, {n_fail} failed, {n_skip} skipped.",
        file=sys.stderr,
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
