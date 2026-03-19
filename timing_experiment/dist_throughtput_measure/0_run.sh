#!/usr/bin/env bash

set -euo pipefail

GPU_LIST="${1:-0,1}"
NUM_GPUS="${2:-2}"
PYTHON_BIN="${3:-python}"

OMP_THREADS="${OMP_NUM_THREADS:-8}"
EPOCHS="${EPOCHS:-5}"
REPEATS="${REPEATS:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29650}"

MODELS_CSV="${MODELS:-TGN}"
DATASETS_CSV="${DATASETS:-LASTFM}"
# BATCH_SIZES_CSV="${BATCH_SIZES:-300, 600, 900, 1200, 1600, 2000, 4000, 6000, 8000}"
BATCH_SIZES_CSV="${BATCH_SIZES:-12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000, 44000, 48000, 52000}"

MEM_DIMS_CSV="${MEM_DIMS:-128,256,512}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$SCRIPT_DIR/logs"
TMP_CONFIG_DIR="$SCRIPT_DIR/tmp_configs"
USER_PREFIX="${LOG_USER_PREFIX:-${USER:-$(id -un)}}"

mkdir -p "$LOG_DIR" "$TMP_CONFIG_DIR"

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"
IFS=',' read -r -a BATCH_SIZES <<< "$BATCH_SIZES_CSV"
IFS=',' read -r -a MEM_DIMS <<< "$MEM_DIMS_CSV"

get_config_path() {
  local model="$1"
  case "$model" in
    TGN|APAN|JODIE|TGAT|DySAT)
      printf '%s/config/%s.yml\n' "$REPO_ROOT" "$model"
      ;;
    *)
      echo "Unsupported model: $model" >&2
      return 1
      ;;
  esac
}

get_dataset_extra_args() {
  local dataset="$1"
  case "$dataset" in
    LASTFM|MOOC)
      printf '%s\n' '--rnd_edim 128'
      ;;
    WIKI|REDDIT)
      printf '\n'
      ;;
    *)
      printf '\n'
      ;;
  esac
}

make_batch_config() {
  local src_cfg="$1"
  local dst_cfg="$2"
  local batch_size="$3"
  local epochs="$4"
  local mem_dim="$5"

  "$PYTHON_BIN" - "$src_cfg" "$dst_cfg" "$batch_size" "$epochs" "$mem_dim" <<'PY'
import sys
import yaml

src, dst, batch_size, epochs, mem_dim = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
with open(src, 'r', encoding='utf-8') as f:
    conf = yaml.safe_load(f)

if 'train' not in conf or not conf['train']:
    raise RuntimeError(f'No train section found in {src}')

conf['train'][0]['batch_size'] = batch_size
conf['train'][0]['epoch'] = epochs

if 'memory' in conf and conf['memory']:
    conf['memory'][0]['dim_out'] = mem_dim

with open(dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(conf, f, sort_keys=False)
PY
}

run_idx=0
for model in "${MODELS[@]}"; do
  config_path="$(get_config_path "$model")"

  for dataset in "${DATASETS[@]}"; do
    extra_args="$(get_dataset_extra_args "$dataset")"

    for mem_dim in "${MEM_DIMS[@]}"; do
      mem_dim="$(echo "$mem_dim" | tr -d ' ')"

      for batch_size in "${BATCH_SIZES[@]}"; do
        batch_size="$(echo "$batch_size" | tr -d ' ')"
        batch_config="$TMP_CONFIG_DIR/${model}_${dataset}_bs${batch_size}_memdim${mem_dim}_ep${EPOCHS}.yml"
        make_batch_config "$config_path" "$batch_config" "$batch_size" "$EPOCHS" "$mem_dim"

        for ((repeat_idx = 1; repeat_idx <= REPEATS; repeat_idx++)); do
          master_port=$((BASE_MASTER_PORT + run_idx))
          log_file="$LOG_DIR/${USER_PREFIX}_${model}_${dataset}_bs${batch_size}_ngpu${NUM_GPUS}_memdim${mem_dim}_ep${EPOCHS}_rep${repeat_idx}.log"

          echo "============================================================"
          echo "[model=${model} dataset=${dataset} batch_size=${batch_size} mem_dim=${mem_dim} num_gpus=${NUM_GPUS} epochs=${EPOCHS} repeat=${repeat_idx}]"
          echo "log: ${log_file}"
          echo "master_port: ${master_port}"
          echo "============================================================"

          cmd=(
            "$PYTHON_BIN" -m torch.distributed.run
            --nproc_per_node "$((NUM_GPUS + 1))"
            --master_addr "$MASTER_ADDR"
            --master_port "$master_port"
            "$REPO_ROOT/train_dist.py"
            --dataset "$dataset"
            --config "$batch_config"
            --num_gpus "$NUM_GPUS"
            --omp_num_threads "$OMP_THREADS"
          )

          if [[ -n "$extra_args" ]]; then
            read -r -a extra_parts <<< "$extra_args"
            cmd+=("${extra_parts[@]}")
          fi

          CUDA_VISIBLE_DEVICES="$GPU_LIST" "${cmd[@]}" 2>&1 | tee "$log_file"
          run_idx=$((run_idx + 1))
        done
      done
    done
  done
done

echo "All logs saved to ${LOG_DIR}"
