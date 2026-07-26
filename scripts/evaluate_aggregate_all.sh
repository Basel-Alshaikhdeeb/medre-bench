#!/bin/bash
# Run both evaluations (aggregate/test 6-class + per-source binary) on every
# aggregate-trained checkpoint found under $OUTPUTS_DIR. Idempotent: skips
# checkpoints whose eval JSON already exists so partial runs can resume.
#
# Usage:
#     OUTPUTS_DIR=outputs SPLIT=test bash scripts/evaluate_aggregate_all.sh
#
# Env vars:
#     OUTPUTS_DIR   root outputs/ tree                     (default: outputs)
#     SPLIT         which split to score on                (default: test)
#     BATCH_SIZE    inference batch size                    (default: 64)
#     SOURCES       comma-sep subset for evaluate-aggregate (default: all 7)
#     FORCE         set to 1 to re-run even if JSON exists (default: unset)
#
# Wrap in a SLURM job with --gres=gpu:1 for GPU acceleration.

set -euo pipefail

OUTPUTS_DIR="${OUTPUTS_DIR:-outputs}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SOURCES="${SOURCES:-}"
FORCE="${FORCE:-}"

if ! command -v medre-bench >/dev/null 2>&1; then
  echo "medre-bench CLI not on PATH — source your venv first" >&2
  exit 1
fi

echo "[eval] outputs_dir=$OUTPUTS_DIR split=$SPLIT batch_size=$BATCH_SIZE"

n_ckpts=0
n_ran_agg=0
n_ran_per_src=0
n_skipped=0

for run_dir in "$OUTPUTS_DIR"/*__aggregate__seed*/; do
  [ -d "$run_dir" ] || continue

  # Newest timestamped subdir (in case there were multiple training attempts).
  ts_dir=$(find "$run_dir" -maxdepth 1 -mindepth 1 -type d | sort | tail -1)
  [ -n "$ts_dir" ] || continue
  ckpt="$ts_dir/checkpoints/best"

  if [ ! -f "$ckpt/model.safetensors" ] && [ ! -f "$ckpt/pytorch_model.bin" ]; then
    echo "  skip $run_dir: no weights in $ckpt"
    continue
  fi
  n_ckpts=$((n_ckpts + 1))

  echo
  echo "=== $(basename "$run_dir") ==="

  # ---- 1) Aggregate/test: 6-class multiclass ---------------------------------
  agg_out="$ts_dir/aggregate_test_eval"
  agg_json="$agg_out/metrics_aggregate_${SPLIT}.json"
  if [ -z "$FORCE" ] && [ -f "$agg_json" ]; then
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

  # ---- 2) Per-source binary --------------------------------------------------
  src_out="$ts_dir/aggregate_eval"
  src_json="$src_out/aggregate_eval_${SPLIT}.json"
  if [ -z "$FORCE" ] && [ -f "$src_json" ]; then
    echo "  skip evaluate-aggregate: $src_json exists"
    n_skipped=$((n_skipped + 1))
  else
    if [ -n "$SOURCES" ]; then
      medre-bench evaluate-aggregate \
        --checkpoint "$ckpt" \
        --split "$SPLIT" \
        --sources "$SOURCES" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$src_out"
    else
      medre-bench evaluate-aggregate \
        --checkpoint "$ckpt" \
        --split "$SPLIT" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$src_out"
    fi
    n_ran_per_src=$((n_ran_per_src + 1))
  fi
done

echo
echo "[eval] done: checkpoints=$n_ckpts, ran_aggregate=$n_ran_agg, ran_per_source=$n_ran_per_src, cached=$n_skipped"
