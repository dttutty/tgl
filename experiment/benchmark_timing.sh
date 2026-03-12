#!/bin/bash
# Benchmark TGN, APAN, JODIE across 6 datasets with pin/nopin modes.
# Usage: bash experiment/benchmark_timing.sh [GPU_ID]
# Logs saved to: experiment/benchmark_timing/

set -e

GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/benchmark_timing"

mkdir -p "$LOG_DIR"

MODELS=("TGN" "APAN" "JODIE")
CONFIGS=("config/TGN.yml" "config/APAN.yml" "config/JODIE.yml")

DATASETS=("LASTFM" "MOOC" "REDDIT" "WIKIPEDIA" "GDELT" "MAG")
# extra args per dataset (index-matched with DATASETS)
EXTRA_ARGS=(
    "--rand_edge_features 128"   # LASTFM
    "--rand_edge_features 128"   # MOOC
    ""                           # REDDIT
    ""                           # WIKIPEDIA
    ""                           # GDELT
    ""                           # MAG
)

PIN_MODES=("nopin" "pin")
PIN_FLAGS=("" "--pin_memory")

for d_idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$d_idx]}"
    extra="${EXTRA_ARGS[$d_idx]}"
    for m_idx in "${!MODELS[@]}"; do
        model="${MODELS[$m_idx]}"
        config="${CONFIGS[$m_idx]}"
        for p_idx in "${!PIN_MODES[@]}"; do
            pin_label="${PIN_MODES[$p_idx]}"
            pin_flag="${PIN_FLAGS[$p_idx]}"
            log_file="$LOG_DIR/${model}_${dataset}_${pin_label}.log"

            echo "============================================================"
            echo "[${model} / ${dataset} / ${pin_label}] -> ${log_file}"
            echo "============================================================"

            python "$REPO_ROOT/train.py" \
                --data "$dataset" \
                --config "$REPO_ROOT/$config" \
                --gpu "$GPU" \
                --model_name "${model}_${dataset}_${pin_label}" \
                $extra $pin_flag \
                2>&1 | tee "$log_file"

            echo ""
        done
    done
done

echo "All logs saved to $LOG_DIR"
