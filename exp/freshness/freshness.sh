#!/bin/bash
# Freshness sweep for delayed memory updates.
# Runs train_non_timing.py with pin_memory enabled and sweeps
# memory_update_delay_batches across datasets and model configs.
#
# Usage:
#   bash exp/freshness/freshness.sh [GPU_ID]

set -uo pipefail

GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$SCRIPT_DIR/logs"
PYTHON_BIN="${PYTHON_BIN:-/home/sqp17/miniconda3/envs/simple_py310/bin/python}"
TMP_CONFIG_DIR="$LOG_DIR/tmp_configs"

mkdir -p "$LOG_DIR"
mkdir -p "$TMP_CONFIG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

MODELS=("TGN" "APAN" "JODIE")
CONFIGS=("config/TGN.yml" "config/APAN.yml" "config/JODIE.yml")

DATASETS=("LASTFM" "MOOC" "REDDIT" "WIKI")
EXTRA_ARGS=(
    "--rand_edge_features 128"
    "--rand_edge_features 128"
    ""
    ""
)

DELAYS=(0 1 2 3 4)
DIM_OUTS=(128 256 512)
REPEATS=20
TARGET_EPOCH=100

make_dim_config() {
    local src_cfg="$1"
    local dst_cfg="$2"
    local dim_out="$3"

    "$PYTHON_BIN" - "$src_cfg" "$dst_cfg" "$dim_out" "$TARGET_EPOCH" <<'PY'
import sys
import yaml

src, dst, dim_out, target_epoch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(src, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f)

if "memory" not in conf or not conf["memory"]:
    raise RuntimeError(f"No memory section found in {src}")
if "train" not in conf or not conf["train"]:
    raise RuntimeError(f"No train section found in {src}")

conf["memory"][0]["dim_out"] = dim_out
conf["train"][0]["epoch"] = target_epoch

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(conf, f, sort_keys=False)
PY
}

echo "Logs: $LOG_DIR"
echo "Python: $PYTHON_BIN"
echo "Epoch per run: $TARGET_EPOCH"
echo "Repeats per experiment: $REPEATS"

for d_idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$d_idx]}"
    extra="${EXTRA_ARGS[$d_idx]}"
    extra_args=()
    if [[ -n "$extra" ]]; then
        read -r -a extra_args <<< "$extra"
    fi

    for m_idx in "${!MODELS[@]}"; do
        model="${MODELS[$m_idx]}"
        config="${CONFIGS[$m_idx]}"
        dim_config_base="$REPO_ROOT/$config"

        for dim_out in "${DIM_OUTS[@]}"; do
            dim_config="$TMP_CONFIG_DIR/${model}_dim${dim_out}.yml"
            make_dim_config "$dim_config_base" "$dim_config" "$dim_out"

            for delay in "${DELAYS[@]}"; do
                for run_id in $(seq 1 "$REPEATS"); do
                    log_file="$LOG_DIR/${model}_${dataset}_dim${dim_out}_delay${delay}_run${run_id}_pin.log"

                    echo "============================================================"
                    echo "[${model} / ${dataset} / dim_out=${dim_out} / delay=${delay} / run=${run_id}/${REPEATS} / pin_memory=true / epoch=${TARGET_EPOCH}]"
                    echo "config=$dim_config"
                    echo "log=$log_file"
                    echo "============================================================"

                    "$PYTHON_BIN" -u "$REPO_ROOT/train_non_timing.py" \
                        --data "$dataset" \
                        --config "$dim_config" \
                        --gpu "$GPU" \
                        --model_name "${model}_${dataset}_dim${dim_out}_delay${delay}_run${run_id}_pin" \
                        --pin_memory \
                        --memory_update_delay_batches "$delay" \
                        "${extra_args[@]}" \
                        2>&1 | tee "$log_file"

                    echo ""
                done
            done
        done
    done
done

echo "All runs finished."
echo "Logs saved to $LOG_DIR"

