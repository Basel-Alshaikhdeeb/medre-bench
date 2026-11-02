#!/usr/bin/env python3
"""Generate SLURM scripts for a (model, dataset, seed) sweep.

Two modes:

* ``--mode per-run`` (default): one ``.sh`` per combo plus ``submit_all.sh``.
  Fine-grained control; one script per `sbatch`. Good when you want to retry
  individual runs or watch logs one at a time.

* ``--mode array``: two SLURM array scripts, ``train_small.slurm`` and
  ``train_large.slurm``. Each embeds the list of combos as a bash array
  indexed by ``SLURM_ARRAY_TASK_ID``, and each task:
    - picks the (model, dataset, seed) for its index,
    - looks up the dataset's max_seq_length from a case statement,
    - skips the combo if ``.../checkpoints/best/model.safetensors`` exists
      (so you can resubmit cheaply after interruptions),
    - invokes ``medre-bench train`` with ``model.max_seq_length=<N>`` plus
      any extra overrides.
  Large-model and small-model arrays get different ``--time`` and
  concurrency caps — the user's words: "short arrays for small models and
  vice versa".

Dataset max_seq_length defaults match the user's measured 95th percentiles:
    euadr=344 ddi=512 chem_dis_gene=512 gad=144 bc5cdr=304
    chemprot=440 drugprot=256 biored=352
Override with ``--max-seq-length-map "euadr=344,ddi=512,..."``.

Model-size grouping: anything matching ``--large-models`` (default list below)
runs in the large array; the rest in the small array. Override with
``--large-models foo,bar,baz`` or ``--large-models ""`` to force all small.

Usage (array mode, re-training the multiTomek paradigm):
    python scripts/generate_slurm_jobs.py --mode array \\
        --models pubmedbert-large,biolinkbert-large,roberta-large,gatortron,\\
pubmedbert,biobert,biolinkbert-base,biomed-roberta,pubmed-roberta,\\
roberta-base,bert-base,scibert,biogpt,gpt2,clinicalbert,bio-clinicalbert \\
        --datasets bc5cdr,biored,chem_dis_gene,chemprot,ddi,drugprot,euadr,gad \\
        --seeds 42,123,456,2024,1337 \\
        --jobs-dir slurm_jobs/multiTomek_retrain \\
        --output-dir /scratch/users/balshaikhdeeb/medre-bench/outputs_multiTomek \\
        --partition hopper --qos iris-hopper \\
        --time-small 02:00:00 --time-large 08:00:00 \\
        --array-concurrency-small 16 --array-concurrency-large 8 \\
        --extra-override training.paradigm=multiTomek \\
        --extra-override training.epochs=5

Then:
    sbatch slurm_jobs/multiTomek_retrain/train_small.slurm
    sbatch slurm_jobs/multiTomek_retrain/train_large.slurm
"""

from __future__ import annotations

import argparse
import shlex
import stat
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset-specific max_seq_length (95th percentile of training tokens).
# Edit DATASET_MAX_SEQ_LENGTH to change defaults, or pass --max-seq-length-map
# on the CLI to override at generation time.
# ---------------------------------------------------------------------------
DATASET_MAX_SEQ_LENGTH: dict[str, int] = {
    "euadr": 344,
    "ddi": 512,
    "chem_dis_gene": 512,
    "gad": 144,
    "bc5cdr": 304,
    "chemprot": 440,
    "drugprot": 256,
    "biored": 352,
}

# Default heuristic — override on the CLI if your sweep uses other large models.
DEFAULT_LARGE_MODELS: list[str] = [
    "pubmedbert-large",
    "biolinkbert-large",
    "roberta-large",
    "gatortron",
]

# ---------------------------------------------------------------------------
# Edit this block to match your HPC site (modules, conda/venv path, etc.).
# Lines are inlined into every generated script; keep it minimal.
# ---------------------------------------------------------------------------
ENV_SETUP = """\
# --- Environment setup (edit for your HPC site) -----------------------------
set -euo pipefail
module load env/development/2024a
module load env/release/2023b
module load env/release/default
module load system/CUDA/12.6.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export MASTER_PORT=$(expr 20000 + $SLURM_JOB_ID % 10000)
export HF_HOME=~/scratch/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd "${MEDRE_ROOT:-$HOME/medre-bench}"
source "${MEDRE_VENV:-$HOME/medre-bench/.venv}/bin/activate"
# ---------------------------------------------------------------------------
"""

JOB_TEMPLATE = """\
#!/bin/sh -l
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/%x_%j.out
#SBATCH --error={logs_dir}/%x_%j.err
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
{account_line}{qos_line}{nodelist_line}
{env_setup}
echo "[job] host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date -Is)"
nvidia-smi || true

{run_cmd}

echo "[job] end=$(date -Is) status=$?"
"""


def _maybe_line(flag: str, value: str | None) -> str:
    """Return an #SBATCH line if value is set, else empty string."""
    return f"#SBATCH {flag}={value}\n" if value else ""


def _build_run_cmd(
    *,
    model: str,
    dataset: str,
    seed: int,
    experiment_name: str,
    output_dir: str,
    num_gpus: int,
    deepspeed_config: str | None,
    extra_overrides: list[str],
    max_seq_length_map: dict[str, int] | None = None,
) -> str:
    """Construct the python/torchrun command for a single experiment."""
    overrides: list[str] = [
        f"training.seed={seed}",
        f"experiment_name={experiment_name}",
    ]
    if max_seq_length_map and dataset in max_seq_length_map:
        overrides.append(f"model.max_seq_length={max_seq_length_map[dataset]}")
    if deepspeed_config:
        overrides.append(f'training.deepspeed="{deepspeed_config}"')
    overrides.extend(extra_overrides)

    overrides_str = " ".join(shlex.quote(o) for o in overrides)

    if num_gpus > 1 or deepspeed_config:
        # Distributed launch via torchrun (works for DDP and DeepSpeed).
        return (
            f"torchrun --standalone --nproc_per_node={num_gpus} \\\n"
            f"    scripts/distributed_train.py \\\n"
            f"    --model {model} \\\n"
            f"    --dataset {dataset} \\\n"
            f"    --output-dir {output_dir} \\\n"
            f"    {overrides_str}"
        )
    return (
        f"medre-bench train \\\n"
        f"    --model {model} \\\n"
        f"    --dataset {dataset} \\\n"
        f"    --output-dir {output_dir} \\\n"
        f"    {overrides_str}"
    )


ARRAY_TEMPLATE = """\
#!/bin/sh -l
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/%x_%A_%a.out
#SBATCH --error={logs_dir}/%x_%A_%a.err
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --array=0-{last_idx}%{concurrency}
{account_line}{qos_line}{nodelist_line}
{env_setup}
echo "[job] host=$(hostname) task=$SLURM_ARRAY_TASK_ID gpus=$CUDA_VISIBLE_DEVICES start=$(date -Is)"
nvidia-smi || true

# One line per combo: "<model> <dataset> <seed>"
COMBOS=(
{combos_block}
)

TRIPLE="${{COMBOS[$SLURM_ARRAY_TASK_ID]}}"
if [ -z "$TRIPLE" ]; then
    echo "no combo at index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
MODEL=$(echo "$TRIPLE" | awk '{{print $1}}')
DATASET=$(echo "$TRIPLE" | awk '{{print $2}}')
SEED=$(echo "$TRIPLE" | awk '{{print $3}}')

# Dataset -> max_seq_length (95th percentile of training tokens).
case "$DATASET" in
{case_block}
    *) MAX_LEN={default_max_len} ;;
esac

EXPERIMENT_NAME="${{MODEL}}__${{DATASET}}__seed${{SEED}}"
echo "[task $SLURM_ARRAY_TASK_ID] $EXPERIMENT_NAME  max_seq_length=$MAX_LEN"

# Idempotence: if the best checkpoint already exists, skip. Lets you resubmit
# the array after interruptions without re-training completed combos.
if [ -z "${{FORCE:-}}" ] && \\
   find "{output_dir}/$EXPERIMENT_NAME" -mindepth 1 -maxdepth 5 \\
        -name model.safetensors -path '*/checkpoints/best/*' 2>/dev/null \\
   | grep -q .; then
    echo "  skip: best checkpoint already present"
    exit 0
fi

{run_cmd_prefix} \\
    model.max_seq_length=$MAX_LEN \\
    training.seed=$SEED \\
    experiment_name=$EXPERIMENT_NAME{extra_overrides_block}

echo "[task $SLURM_ARRAY_TASK_ID] end=$(date -Is) status=$?"
"""


def _is_large(model: str, large_models: set[str]) -> bool:
    """Decide whether a model goes in the large-model array."""
    if model in large_models:
        return True
    # Fallback: anything with "large" in the name is treated as large.
    return "large" in model.lower()


def _build_combos_block(combos: list[tuple[str, str, int]]) -> str:
    return "\n".join(f'    "{m} {d} {s}"' for m, d, s in combos)


def _build_case_block(max_seq_map: dict[str, int]) -> str:
    return "\n".join(f"    {ds}) MAX_LEN={ml} ;;" for ds, ml in max_seq_map.items())


def _build_run_cmd_prefix(
    *,
    output_dir: str,
    num_gpus: int,
    deepspeed_config: str | None,
) -> str:
    """Base command minus the per-task overrides (model/dataset come from env)."""
    if num_gpus > 1 or deepspeed_config:
        head = (
            f"torchrun --standalone --nproc_per_node={num_gpus} \\\n"
            f"    scripts/distributed_train.py \\\n"
            f'    --model "$MODEL" \\\n'
            f'    --dataset "$DATASET" \\\n'
            f'    --output-dir "{output_dir}"'
        )
    else:
        head = (
            f"medre-bench train \\\n"
            f'    --model "$MODEL" \\\n'
            f'    --dataset "$DATASET" \\\n'
            f'    --output-dir "{output_dir}"'
        )
    if deepspeed_config:
        head += f' \\\n    training.deepspeed="{deepspeed_config}"'
    return head


def generate_array(
    *,
    models: list[str],
    datasets: list[str],
    seeds: list[int],
    jobs_dir: Path,
    output_dir: str,
    partition: str,
    num_gpus: int,
    cpus: int,
    mem: str,
    account: str | None,
    qos: str | None,
    nodelist: str | None,
    deepspeed_config: str | None,
    extra_overrides: list[str],
    large_models: list[str],
    max_seq_length_map: dict[str, int],
    time_small: str,
    time_large: str,
    array_concurrency_small: int,
    array_concurrency_large: int,
) -> list[Path]:
    """Write train_small.slurm and train_large.slurm (array scripts) + README."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = jobs_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    large_set = set(large_models)
    small_combos: list[tuple[str, str, int]] = []
    large_combos: list[tuple[str, str, int]] = []
    for model in models:
        bucket = large_combos if _is_large(model, large_set) else small_combos
        for dataset in datasets:
            for seed in seeds:
                bucket.append((model, dataset, seed))

    extra_block = (
        " \\\n    " + " \\\n    ".join(shlex.quote(o) for o in extra_overrides)
        if extra_overrides else ""
    )
    case_block = _build_case_block(max_seq_length_map)
    default_max_len = max(max_seq_length_map.values()) if max_seq_length_map else 512
    run_cmd_prefix = _build_run_cmd_prefix(
        output_dir=output_dir,
        num_gpus=num_gpus,
        deepspeed_config=deepspeed_config,
    )

    written: list[Path] = []
    for name, combos, time, concurrency in (
        ("train_small", small_combos, time_small, array_concurrency_small),
        ("train_large", large_combos, time_large, array_concurrency_large),
    ):
        if not combos:
            print(f"skip {name}.slurm: no combos in this bucket")
            continue
        content = ARRAY_TEMPLATE.format(
            job_name=name,
            logs_dir=str(logs_dir),
            time=time,
            partition=partition,
            num_gpus=num_gpus,
            cpus=cpus,
            mem=mem,
            last_idx=len(combos) - 1,
            concurrency=max(1, concurrency),
            account_line=_maybe_line("--account", account),
            qos_line=_maybe_line("--qos", qos),
            nodelist_line=_maybe_line("--nodelist", nodelist),
            env_setup=ENV_SETUP,
            combos_block=_build_combos_block(combos),
            case_block=case_block,
            default_max_len=default_max_len,
            output_dir=output_dir,
            run_cmd_prefix=run_cmd_prefix,
            extra_overrides_block=extra_block,
        )
        path = jobs_dir / f"{name}.slurm"
        path.write_text(content)
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        written.append(path)

    # A README listing every combo per bucket, so users can map array
    # indices back to (model, dataset, seed) when triaging failures.
    readme_lines = ["# Combos", ""]
    for name, combos in (("train_small", small_combos), ("train_large", large_combos)):
        if not combos:
            continue
        readme_lines.append(f"## {name}.slurm ({len(combos)} tasks)")
        readme_lines.append("")
        readme_lines.append("| idx | model | dataset | seed |")
        readme_lines.append("|---|---|---|---|")
        for i, (m, d, s) in enumerate(combos):
            readme_lines.append(f"| {i} | {m} | {d} | {s} |")
        readme_lines.append("")
    (jobs_dir / "combos.md").write_text("\n".join(readme_lines))

    return written


def generate(
    *,
    models: list[str],
    datasets: list[str],
    seeds: list[int],
    jobs_dir: Path,
    output_dir: str,
    time: str,
    partition: str,
    num_gpus: int,
    cpus: int,
    mem: str,
    account: str | None,
    qos: str | None,
    nodelist: str | None,
    deepspeed_config: str | None,
    extra_overrides: list[str],
    max_seq_length_map: dict[str, int] | None = None,
) -> list[Path]:
    """Write one .sh per combo plus submit_all.sh. Returns list of script paths."""
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = jobs_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    written: list[Path] = []
    for model in models:
        for dataset in datasets:
            for seed in seeds:
                experiment_name = f"{model}__{dataset}__seed{seed}"
                job_name = experiment_name
                run_cmd = _build_run_cmd(
                    model=model,
                    dataset=dataset,
                    seed=seed,
                    experiment_name=experiment_name,
                    output_dir=output_dir,
                    num_gpus=num_gpus,
                    deepspeed_config=deepspeed_config,
                    extra_overrides=extra_overrides,
                    max_seq_length_map=max_seq_length_map,
                )
                content = JOB_TEMPLATE.format(
                    job_name=job_name,
                    logs_dir=str(logs_dir),
                    time=time,
                    partition=partition,
                    num_gpus=num_gpus,
                    cpus=cpus,
                    mem=mem,
                    account_line=_maybe_line("--account", account),
                    qos_line=_maybe_line("--qos", qos),
                    nodelist_line=_maybe_line("--nodelist", nodelist),
                    env_setup=ENV_SETUP,
                    run_cmd=run_cmd,
                )
                path = jobs_dir / f"{experiment_name}.sh"
                path.write_text(content)
                path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                written.append(path)

    submit_all = jobs_dir / "submit_all.sh"
    submit_lines = ["#!/bin/bash", "set -e", f'cd "$(dirname "$0")"', ""]
    submit_lines += [f"sbatch {p.name}" for p in written]
    submit_all.write_text("\n".join(submit_lines) + "\n")
    submit_all.chmod(submit_all.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return written


def _parse_max_seq_map(spec: str | None) -> dict[str, int]:
    """Parse 'ds=N,ds=N,...' overlaid on top of DATASET_MAX_SEQ_LENGTH."""
    out = dict(DATASET_MAX_SEQ_LENGTH)
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"--max-seq-length-map entry {part!r} must be of the form dataset=INT"
            )
        ds, val = part.split("=", 1)
        out[ds.strip()] = int(val.strip())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", required=True, help="Comma-separated model registry keys")
    p.add_argument("--datasets", required=True, help="Comma-separated dataset registry keys")
    p.add_argument("--seeds", default="42", help="Comma-separated integer seeds")
    p.add_argument("--jobs-dir", required=True, help="Directory to write generated .sh files into")
    p.add_argument("--output-dir", default="outputs", help="Training output_dir passed to medre-bench")

    p.add_argument("--mode", choices=["per-run", "array"], default="per-run",
                   help="per-run (default): one .sh per combo + submit_all.sh. "
                        "array: train_small.slurm + train_large.slurm with SLURM job arrays.")

    # SLURM resources
    p.add_argument("--time", default="12:00:00",
                   help="[per-run mode] Wall-clock limit (HH:MM:SS)")
    p.add_argument("--time-small", default="02:00:00",
                   help="[array mode] Wall-clock per task in train_small.slurm (default: 02:00:00)")
    p.add_argument("--time-large", default="08:00:00",
                   help="[array mode] Wall-clock per task in train_large.slurm (default: 08:00:00)")
    p.add_argument("--array-concurrency-small", type=int, default=16,
                   help="[array mode] Max concurrent tasks in train_small.slurm (default: 16)")
    p.add_argument("--array-concurrency-large", type=int, default=8,
                   help="[array mode] Max concurrent tasks in train_large.slurm (default: 8)")
    p.add_argument("--partition", required=True, help="SLURM partition")
    p.add_argument("--num-gpus", type=int, default=1, help="GPUs per job (gres=gpu:N)")
    p.add_argument("--cpus", type=int, default=8, help="CPUs per task")
    p.add_argument("--mem", default="64G", help="Memory request")
    p.add_argument("--account", default=None, help="SLURM --account (optional)")
    p.add_argument("--qos", default=None, help="SLURM --qos (optional)")
    p.add_argument("--nodelist", default=None, help="SLURM --nodelist (optional)")

    # Model-size grouping (array mode)
    p.add_argument("--large-models", default=",".join(DEFAULT_LARGE_MODELS),
                   help=f"[array mode] Comma-separated model keys to route to "
                        f"train_large.slurm. Anything containing 'large' is also "
                        f"treated as large. Default: {','.join(DEFAULT_LARGE_MODELS)}. "
                        f"Pass an empty string to force all small.")

    # Dataset-specific max_seq_length
    p.add_argument("--max-seq-length-map", default=None,
                   help="Override dataset max_seq_length values; "
                        "format 'euadr=344,ddi=512,...'. Any dataset not in the "
                        "map falls back to the built-in defaults.")

    # Training knobs
    p.add_argument("--deepspeed-config", default=None,
                   help="Path to DeepSpeed JSON; forces torchrun launch")
    p.add_argument("--extra-override", action="append", default=[],
                   help="Pass-through override (repeatable), e.g. training.epochs=5")

    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    large_models = [m.strip() for m in args.large_models.split(",") if m.strip()]
    max_seq_length_map = _parse_max_seq_map(args.max_seq_length_map)

    if args.mode == "array":
        written = generate_array(
            models=models,
            datasets=datasets,
            seeds=seeds,
            jobs_dir=Path(args.jobs_dir),
            output_dir=args.output_dir,
            partition=args.partition,
            num_gpus=args.num_gpus,
            cpus=args.cpus,
            mem=args.mem,
            account=args.account,
            qos=args.qos,
            nodelist=args.nodelist,
            deepspeed_config=args.deepspeed_config,
            extra_overrides=args.extra_override,
            large_models=large_models,
            max_seq_length_map=max_seq_length_map,
            time_small=args.time_small,
            time_large=args.time_large,
            array_concurrency_small=args.array_concurrency_small,
            array_concurrency_large=args.array_concurrency_large,
        )
        print(f"Wrote {len(written)} array script(s) + combos.md to {args.jobs_dir}")
        for p_ in written:
            print(f"  sbatch {p_}")
        return

    written = generate(
        models=models,
        datasets=datasets,
        seeds=seeds,
        jobs_dir=Path(args.jobs_dir),
        output_dir=args.output_dir,
        time=args.time,
        partition=args.partition,
        num_gpus=args.num_gpus,
        cpus=args.cpus,
        mem=args.mem,
        account=args.account,
        qos=args.qos,
        nodelist=args.nodelist,
        deepspeed_config=args.deepspeed_config,
        extra_overrides=args.extra_override,
        max_seq_length_map=max_seq_length_map,
    )
    print(f"Wrote {len(written)} job scripts to {args.jobs_dir}")
    print(f"Submit with: bash {Path(args.jobs_dir) / 'submit_all.sh'}")


if __name__ == "__main__":
    main()
