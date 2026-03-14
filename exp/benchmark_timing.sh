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

MODELS=("TGN" )
CONFIGS=("config/TGN.yml" )

DATASETS=("WIKI" )
# extra args per dataset (index-matched with DATASETS)
EXTRA_ARGS=(
    "--rand_edge_features 128"   # LASTFM
    "--rand_edge_features 128"   # MOOC
    ""                           # REDDIT
    ""                           # WIKI
    ""                           # GDELT
    ""                           # MAG
)

PIN_MODES=("nopin" "pin")
PIN_FLAGS=("" "--pin_memory")
DIM_OUTS=(128 256 384 512)

TMP_CONFIG_DIR="$LOG_DIR/tmp_configs"
mkdir -p "$TMP_CONFIG_DIR"

make_dim_config() {
    local src_cfg="$1"
    local dst_cfg="$2"
    local dim_out="$3"
    python - "$src_cfg" "$dst_cfg" "$dim_out" <<'PY'
import sys
import yaml

src, dst, dim_out = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f)

if "memory" not in conf or not conf["memory"]:
    raise RuntimeError(f"No memory section found in {src}")

conf["memory"][0]["dim_out"] = dim_out

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(conf, f, sort_keys=False)
PY
}

for d_idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$d_idx]}"
    extra="${EXTRA_ARGS[$d_idx]}"
    for m_idx in "${!MODELS[@]}"; do
        model="${MODELS[$m_idx]}"
        config="${CONFIGS[$m_idx]}"
        for dim_out in "${DIM_OUTS[@]}"; do
            src_config="$REPO_ROOT/$config"
            dim_config="$TMP_CONFIG_DIR/${model}_dim${dim_out}.yml"
            make_dim_config "$src_config" "$dim_config" "$dim_out"

            for p_idx in "${!PIN_MODES[@]}"; do
                pin_label="${PIN_MODES[$p_idx]}"
                pin_flag="${PIN_FLAGS[$p_idx]}"
                log_file="$LOG_DIR/${model}_${dataset}_${pin_label}_dim${dim_out}.log"

                echo "============================================================"
                echo "[${model} / ${dataset} / ${pin_label} / dim_out=${dim_out}] -> ${log_file}"
                echo "============================================================"

                python "$REPO_ROOT/train.py" \
                    --data "$dataset" \
                    --config "$dim_config" \
                    --gpu "$GPU" \
                    --model_name "${model}_${dataset}_${pin_label}_dim${dim_out}" \
                    $extra $pin_flag \
                    2>&1 | tee "$log_file"

                echo ""
            done
        done
    done
done

echo "All logs saved to $LOG_DIR"
