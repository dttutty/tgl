#!/usr/bin/env bash

set -euo pipefail

GPU_LIST="${1:-0,1}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"

if [[ $# -ge 2 && -n "${2:-}" ]]; then
  NUM_GPUS="$2"
else
  NUM_GPUS="${#GPU_IDS[@]}"
fi

if [[ $# -ge 3 && -n "${3:-}" ]]; then
  PYTHON_BIN="$3"
elif [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python interpreter not found." >&2
  exit 1
fi

if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN")"
fi

OMP_THREADS="${OMP_NUM_THREADS:-8}"
MODEL="${MODEL:-TGN}"
DATASET="${DATASET:-LASTFM}"
EPOCHS="${EPOCHS:-100}"
REPEATS="${REPEATS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4000}"
MEM_DIMS_CSV="${MEM_DIMS:-128,256,512}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29750}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/${MODEL}.yml}"
LOG_ROOT="$SCRIPT_DIR/log"
LOG_DIR="$LOG_ROOT/$RUN_TAG"
TMP_CONFIG_DIR="$SCRIPT_DIR/tmp_configs/$RUN_TAG"

mkdir -p "$LOG_ROOT" "$LOG_DIR" "$TMP_CONFIG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "NUM_GPUS must be >= 1, got: $NUM_GPUS" >&2
  exit 1
fi

if [[ "${#GPU_IDS[@]}" -lt "$NUM_GPUS" ]]; then
  echo "GPU_LIST provides ${#GPU_IDS[@]} GPU(s), but NUM_GPUS=$NUM_GPUS" >&2
  exit 1
fi

IFS=',' read -r -a MEM_DIMS <<< "$MEM_DIMS_CSV"

get_dataset_extra_args() {
  local dataset="$1"
  case "$dataset" in
    LASTFM|MOOC)
      printf '%s\n' '--rnd_edim 128'
      ;;
    *)
      printf '\n'
      ;;
  esac
}

make_experiment_config() {
  local src_cfg="$1"
  local dst_cfg="$2"
  local batch_size="$3"
  local epochs="$4"
  local mem_dim="$5"

  "$PYTHON_BIN" - "$src_cfg" "$dst_cfg" "$batch_size" "$epochs" "$mem_dim" <<'PY'
import sys
import yaml

src, dst, batch_size, epochs, mem_dim = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
with open(src, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f)

if "train" not in conf or not conf["train"]:
    raise RuntimeError(f"No train section found in {src}")
if "memory" not in conf or not conf["memory"]:
    raise RuntimeError(f"No memory section found in {src}")

conf["train"][0]["batch_size"] = batch_size
conf["train"][0]["epoch"] = epochs
conf["memory"][0]["dim_out"] = mem_dim

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(conf, f, sort_keys=False)
PY
}

extract_test_ap() {
  local log_file="$1"

  "$PYTHON_BIN" - "$log_file" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"test ap:([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
print(matches[-1] if matches else "NA")
PY
}

extra_args="$(get_dataset_extra_args "$DATASET")"
total_runs=$((REPEATS * ${#MEM_DIMS[@]}))
run_idx=0

echo "Log root: $LOG_ROOT"
echo "Logs: $LOG_DIR"
echo "Tmp configs: $TMP_CONFIG_DIR"
echo "Python: $PYTHON_BIN"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "GPUs: $GPU_LIST"
echo "NUM_GPUS: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Repeats: $REPEATS"
echo "Mem dims: ${MEM_DIMS[*]}"
echo "OMP threads: $OMP_THREADS"
echo "Total runs: $total_runs"

for raw_mem_dim in "${MEM_DIMS[@]}"; do
  mem_dim="${raw_mem_dim//[[:space:]]/}"
  batch_config="$TMP_CONFIG_DIR/${MODEL}_${DATASET}_memdim${mem_dim}_ep${EPOCHS}_bs${BATCH_SIZE}.yml"
  make_experiment_config "$CONFIG_PATH" "$batch_config" "$BATCH_SIZE" "$EPOCHS" "$mem_dim"

  for ((repeat_idx = 1; repeat_idx <= REPEATS; repeat_idx++)); do
    master_port=$((BASE_MASTER_PORT + run_idx))
    log_file="$LOG_DIR/${MODEL}_${DATASET}_ngpu${NUM_GPUS}_memdim${mem_dim}_ep${EPOCHS}_rep${repeat_idx}.log"

    echo "============================================================"
    echo "[run $((run_idx + 1))/$total_runs] model=${MODEL} dataset=${DATASET} mem_dim=${mem_dim} repeat=${repeat_idx}/${REPEATS}"
    echo "config: ${batch_config}"
    echo "log: ${log_file}"
    echo "master_port: ${master_port}"
    echo "============================================================"

    cmd=(
      "$PYTHON_BIN" -u -m torch.distributed.run
      --nproc_per_node "$((NUM_GPUS + 1))"
      --master_addr "$MASTER_ADDR"
      --master_port "$master_port"
      "$REPO_ROOT/train_dist.py"
      --dataset "$DATASET"
      --config "$batch_config"
      --num_gpus "$NUM_GPUS"
      --omp_num_threads "$OMP_THREADS"
    )

    if [[ -n "$extra_args" ]]; then
      read -r -a extra_parts <<< "$extra_args"
      cmd+=("${extra_parts[@]}")
    fi

    CUDA_VISIBLE_DEVICES="$GPU_LIST" "${cmd[@]}" 2>&1 | tee "$log_file"

    test_ap="$(extract_test_ap "$log_file")"
    echo "[RESULT] mem_dim=${mem_dim} repeat=${repeat_idx} test_ap=${test_ap}"

    run_idx=$((run_idx + 1))
  done
done

echo "All logs saved to ${LOG_DIR}"
echo "Parse with: ${PYTHON_BIN} ${SCRIPT_DIR}/parse.py --log_dir ${LOG_DIR}"
