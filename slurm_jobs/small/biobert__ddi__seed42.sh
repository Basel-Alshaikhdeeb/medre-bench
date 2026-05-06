#!/bin/sh -l
#SBATCH --job-name=biobert__ddi__seed42
#SBATCH --output=slurm_jobs/small/logs/%x_%j.out
#SBATCH --error=slurm_jobs/small/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --qos=iris-hopper

# --- Environment setup (edit for your HPC site) -----------------------------
set -euo pipefail
module load env/development/2024a
module load env/development/2025a
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

echo "[job] host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date -Is)"
nvidia-smi || true

medre-bench train \
    --model biobert \
    --dataset ddi \
    --output-dir /scratch/users/basel.alshaikhdeeb/medre-bench/outputs \
    training.seed=42 experiment_name=biobert__ddi__seed42 training.epochs=10 training.batch_size=16 training.save_checkpoints=false dataset.cleaning_strategy=tomek dataset.balance_train=true training.learning_rate=2e-5 training.warmup_ratio=0.1 training.weight_decay=0.01 training.early_stopping_patience=3

echo "[job] end=$(date -Is) status=$?"
