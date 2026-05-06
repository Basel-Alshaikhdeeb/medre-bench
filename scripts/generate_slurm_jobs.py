#!/usr/bin/env python3
"""Generate one SLURM job script per (model, dataset, seed) combination.

Why per-experiment scripts: a single sweep job is fragile -- one slow run drains
the whole wall-clock budget. Splitting into independent jobs lets each combo get
its own time limit, retry independently, and run in parallel on the cluster.

Usage:
    python scripts/generate_slurm_jobs.py \
        --models pubmedbert,biobert \
        --datasets chemprot,bc5cdr \
        --seeds 42,123,456 \
        --jobs-dir slurm_jobs/sweep_v1 \
        --time 12:00:00 \
        --partition gpu \
        --num-gpus 1 \
        --extra-override training.epochs=10 \
        --extra-override training.batch_size=16

Then submit:
    bash slurm_jobs/sweep_v1/submit_all.sh
"""

from __future__ import annotations

import argparse
import shlex
import stat
from pathlib import Path

# ---------------------------------------------------------------------------
# Edit this block to match your HPC site (modules, conda/venv path, etc.).
# Lines are inlined into every generated script; keep it minimal.
# ---------------------------------------------------------------------------
ENV_SETUP = """\
# --- Environment setup (edit for your HPC site) -----------------------------
set -euo pipefail
module purge || true
# module load cuda/12.1 || true
# module load gcc/11.2 || true
source "${MEDRE_VENV:-$HOME/medre-bench/.venv}/bin/activate"
cd "${MEDRE_ROOT:-$HOME/medre-bench}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# ---------------------------------------------------------------------------
"""

JOB_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={logs_dir}/%x_%j.out
#SBATCH --error={logs_dir}/%x_%j.err
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --qos=iris-hopper
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
) -> str:
    """Construct the python/torchrun command for a single experiment."""
    overrides: list[str] = [
        f"training.seed={seed}",
        f"experiment_name={experiment_name}",
    ]
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", required=True, help="Comma-separated model registry keys")
    p.add_argument("--datasets", required=True, help="Comma-separated dataset registry keys")
    p.add_argument("--seeds", default="42", help="Comma-separated integer seeds")
    p.add_argument("--jobs-dir", required=True, help="Directory to write generated .sh files into")
    p.add_argument("--output-dir", default="outputs", help="Training output_dir passed to medre-bench")

    # SLURM resources
    p.add_argument("--time", default="12:00:00", help="Wall-clock limit (HH:MM:SS)")
    p.add_argument("--partition", required=True, help="SLURM partition")
    p.add_argument("--num-gpus", type=int, default=1, help="GPUs per job (gres=gpu:N)")
    p.add_argument("--cpus", type=int, default=8, help="CPUs per task")
    p.add_argument("--mem", default="64G", help="Memory request")
    p.add_argument("--account", default=None, help="SLURM --account (optional)")
    p.add_argument("--qos", default=None, help="SLURM --qos (optional)")
    p.add_argument("--nodelist", default=None, help="SLURM --nodelist (optional)")

    # Training knobs
    p.add_argument("--deepspeed-config", default=None,
                   help="Path to DeepSpeed JSON; forces torchrun launch")
    p.add_argument("--extra-override", action="append", default=[],
                   help="Pass-through override (repeatable), e.g. training.epochs=5")

    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

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
    )
    print(f"Wrote {len(written)} job scripts to {args.jobs_dir}")
    print(f"Submit with: bash {Path(args.jobs_dir) / 'submit_all.sh'}")


if __name__ == "__main__":
    main()
