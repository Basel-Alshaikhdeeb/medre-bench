#!/bin/bash
# SLURM array template for sharded aggregate-model evaluation.
#
# Distributes every outputs/<model>__aggregate__seed* checkpoint across
# --array=0-N tasks. Each task processes 1/N of the checkpoints on its own
# GPU, running BOTH `medre-bench evaluate --dataset aggregate` (6-class
# multiclass) and `medre-bench evaluate-aggregate` (per-source binary) for
# every checkpoint it owns. Idempotent per-file skip so partial runs resume.
#
# Env vars (override with e.g. `sbatch --export=ALL,BATCH_SIZE=128 ...`):
#     OUTPUTS_DIR   root containing the aggregate run dirs   (default: outputs)
#     SPLIT         which split to score on                  (default: test)
#     BATCH_SIZE    inference batch size                     (default: 64)
#     FORCE         set to 1 to re-run even if JSON exists   (default: unset)
#     MEDRE_ROOT    repo root                                (default: $HOME/medre-bench)
#     MEDRE_VENV    venv path                                (default: $MEDRE_ROOT/.venv)
#
# Submit:
#     sbatch scripts/aggregate_eval_slurm.sh

#SBATCH --job-name=aggregate_eval
#SBATCH --output=slurm_jobs/aggregate_eval/logs/%x_%A_%a.out
#SBATCH --error=slurm_jobs/aggregate_eval/logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=iris-hopper

set -euo pipefail

# --- Environment (edit for your site if needed) -----------------------------
module load env/development/2024a
module load env/release/2023b
module load env/release/default
module load system/CUDA/12.6.0

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export HF_HOME=~/scratch/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd "${MEDRE_ROOT:-$HOME/medre-bench}"
source "${MEDRE_VENV:-$HOME/medre-bench/.venv}/bin/activate"

mkdir -p slurm_jobs/aggregate_eval/logs

echo "[job] host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"
nvidia-smi || true

# --- Config ------------------------------------------------------------------
OUTPUTS_DIR="${OUTPUTS_DIR:-outputs}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-64}"

NSHARDS=${SLURM_ARRAY_TASK_COUNT}
SHARD=${SLURM_ARRAY_TASK_ID}

TMP_ROOT="${TMPDIR:-/tmp}/aggregate_eval_shards"
mkdir -p "$TMP_ROOT"

# --- Discover + shard the checkpoint list ------------------------------------
# Every dir shaped outputs/<model>__aggregate__seed<N>/ is one unit of work.
SHARD_LIST="$TMP_ROOT/shard_${SLURM_ARRAY_JOB_ID}_${SHARD}.txt"
find "$OUTPUTS_DIR" -maxdepth 1 -mindepth 1 -type d -name '*__aggregate__seed*' \
  | sort | awk "NR % $NSHARDS == $SHARD" > "$SHARD_LIST"

n_ckpts=$(wc -l < "$SHARD_LIST" | tr -d ' ')
echo "[job] shard=$SHARD/$NSHARDS has $n_ckpts checkpoint(s) to process"
if [ "$n_ckpts" -eq 0 ]; then
  echo "[job] nothing to do for shard $SHARD; exiting cleanly"
  exit 0
fi
echo "[job] checkpoint list:"
sed 's/^/         /' "$SHARD_LIST"
echo

# --- Process each checkpoint -------------------------------------------------
n_ran_agg=0
n_ran_per_src=0
n_skipped=0
n_missing_weights=0

while IFS= read -r run_dir; do
  [ -d "$run_dir" ] || continue

  # Newest timestamped subdir (in case of multiple training attempts).
  ts_dir=$(find "$run_dir" -maxdepth 1 -mindepth 1 -type d ! -name '_*' | sort | tail -1)
  if [ -z "$ts_dir" ]; then
    echo "  skip $(basename "$run_dir"): no timestamped subdir"
    continue
  fi
  ckpt="$ts_dir/checkpoints/best"

  if [ ! -f "$ckpt/model.safetensors" ] && [ ! -f "$ckpt/pytorch_model.bin" ]; then
    echo "  skip $(basename "$run_dir"): no weights in $ckpt"
    n_missing_weights=$((n_missing_weights + 1))
    continue
  fi

  echo
  echo "=== $(basename "$run_dir") ==="

  # 1) Aggregate/test (6-class multiclass on aggregate corpus test split)
  agg_out="$ts_dir/aggregate_test_eval"
  agg_json="$agg_out/metrics_aggregate_${SPLIT}.json"
  if [ -z "${FORCE:-}" ] && [ -f "$agg_json" ]; then
    echo "  skip evaluate: $agg_json exists"
    n_skipped=$((n_skipped + 1))
  else
    medre-bench evaluate \
      --checkpoint "$ckpt" \
      --dataset aggregate \
      --split "$SPLIT" \
      --output-dir "$agg_out"
    n_ran_agg=$((n_ran_agg + 1))
  fi

  # 2) Per-source binary (each source's own test split at binary granularity)
  src_out="$ts_dir/aggregate_eval"
  src_json="$src_out/aggregate_eval_${SPLIT}.json"
  if [ -z "${FORCE:-}" ] && [ -f "$src_json" ]; then
    echo "  skip evaluate-aggregate: $src_json exists"
    n_skipped=$((n_skipped + 1))
  else
    medre-bench evaluate-aggregate \
      --checkpoint "$ckpt" \
      --split "$SPLIT" \
      --batch-size "$BATCH_SIZE" \
      --output-dir "$src_out"
    n_ran_per_src=$((n_ran_per_src + 1))
  fi
done < "$SHARD_LIST"

echo
echo "[job] shard=$SHARD done at $(date -Is)"
echo "[job]   ran_evaluate=$n_ran_agg"
echo "[job]   ran_evaluate_aggregate=$n_ran_per_src"
echo "[job]   cached=$n_skipped"
echo "[job]   missing_weights=$n_missing_weights"
