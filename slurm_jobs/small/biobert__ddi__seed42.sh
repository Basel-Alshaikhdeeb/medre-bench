#!/bin/bash
#SBATCH --job-name=biobert__ddi__seed42
#SBATCH --output=slurm_jobs/small/logs/%x_%j.out
#SBATCH --error=slurm_jobs/small/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

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

echo "[job] host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date -Is)"
nvidia-smi || true

medre-bench train \
    --model biobert \
    --dataset ddi \
    --output-dir outputs \
    training.seed=42 experiment_name=biobert__ddi__seed42

echo "[job] end=$(date -Is) status=$?"
