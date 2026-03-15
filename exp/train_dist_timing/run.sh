#!/usr/bin/env bash
# Distributed timing benchmark runner for train_dist_timing.py.
#
# Usage examples:
#   bash exp/train_dist_timing/run.sh
#   bash exp/train_dist_timing/run.sh "0,1" 2 /home/sqp17/miniforge3/envs/simple_py310/bin/python
#
# Positional args:
#   1) CUDA_VISIBLE_DEVICES list (default: 0,1)
#   2) num_gpus workers (default: 2)
#   3) python executable (default: python)
#
# Notes:
# - train_dist_timing.py currently always uses pinned buffers internally,
#   so pin label in log filenames is for consistency with benchmark_timing naming.

set -euo pipefail

GPU_LIST="${1:-0,1}"
NUM_GPUS="${2:-2}"
PYTHON_BIN="${3:-python}"
OMP_THREADS="${OMP_NUM_THREADS:-8}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$SCRIPT_DIR/logs"
TMP_CONFIG_DIR="$SCRIPT_DIR/tmp_configs"

mkdir -p "$LOG_DIR" "$TMP_CONFIG_DIR"

MODELS=("TGN" "APAN" "JODIE")
CONFIGS=("config/TGN.yml" "config/APAN.yml" "config/JODIE.yml")

DATASETS=("WIKI" "LASTFM" "MOOC" "REDDIT")
EXTRA_ARGS=(
  ""                      # WIKI
  "--rnd_edim 128"        # LASTFM
  "--rnd_edim 128"        # MOOC
  ""                      # REDDIT
)

# Naming compatibility with benchmark_timing.
PIN_MODES=("pin")
DIM_OUTS=(128 256 384 512)

MASTER_ADDR="127.0.0.1"
BASE_MASTER_PORT="29550"

make_dim_config() {
  local src_cfg="$1"
  local dst_cfg="$2"
  local dim_out="$3"
  "$PYTHON_BIN" - "$src_cfg" "$dst_cfg" "$dim_out" <<'PY'
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

run_idx=0
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

      for pin_label in "${PIN_MODES[@]}"; do
        log_file="$LOG_DIR/${model}_${dataset}_${pin_label}_dim${dim_out}.log"
        master_port=$((BASE_MASTER_PORT + run_idx))

        echo "============================================================"
        echo "[${model} / ${dataset} / ${pin_label} / dim_out=${dim_out}]"
        echo "log: ${log_file}"
        echo "master_port: ${master_port}"
        echo "============================================================"

        cmd=(
          "$PYTHON_BIN" -m torch.distributed.run
          --nproc_per_node "$((NUM_GPUS + 1))"
          --master_addr "$MASTER_ADDR"
          --master_port "$master_port"
          "$REPO_ROOT/train_dist_timing.py"
          --dataset "$dataset"
          --config "$dim_config"
          --num_gpus "$NUM_GPUS"
          --omp_num_threads "$OMP_THREADS"
        )

        if [[ -n "$extra" ]]; then
          # shellcheck disable=SC2206
          extra_parts=($extra)
          cmd+=("${extra_parts[@]}")
        fi

        CUDA_VISIBLE_DEVICES="$GPU_LIST" "${cmd[@]}" 2>&1 | tee "$log_file"
        run_idx=$((run_idx + 1))
      done
    done
  done
done

echo "All logs saved to ${LOG_DIR}"
