#!/usr/bin/env bash

set -euo pipefail

SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_SOURCE")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "$SCRIPT_SOURCE")"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PAIR_RUNNER="$REPO_ROOT/scripts/run_on_gpu_pairs.py"
DATASET_DEFAULTS_SH="$REPO_ROOT/DATA/dataset_defaults.sh"
source "$DATASET_DEFAULTS_SH"
usage() {
  cat <<'EOF'
Usage:
  bash third_party/tgl/run_tgl_grid.sh
  bash third_party/tgl/run_tgl_grid.sh --gpus 0,1,2,3,4,5,6,7
  GPU_IDS=0,1,2,3,4,5,6,7 bash third_party/tgl/run_tgl_grid.sh

This script emits one TGL job per (model, dataset, seed) and lets the
top-level GPU-pair scheduler place them across adjacent GPU pairs.

Environment overrides:
  GPU_IDS=0,1,2,3,4,5,6,7
  MODELS="tgn dyrep jodie apan"
  DATASETS="LASTFM MOOC REDDIT WIKIPEDIA"
  SEEDS="0 1 2 3 4"
  MACRO_BATCH_SIZE=<override dataset default>
  EPOCHS=100
  STABLE_MODE=true
  RUN_ROOT=/abs/path/to/third_party/tgl/seed_sweeps
  MAX_CONCURRENT_JOBS=4
  MAX_JOBS_PER_GPU_PAIR=1
  PAIR_COOLDOWN_SECS=0

Modes:
  --emit-jobs       Print one scheduler job per (model, dataset, seed)
  --run-seed N      Execute exactly one seed job
  --summarize-only  Rebuild results.tsv / summary.txt from seed logs

Notes:
  STABLE_MODE is intentionally restricted to true for this workflow.
  Each scheduled TGL training job uses exactly 2 visible GPUs.
EOF
}

shell_join() {
  local parts=()
  local arg

  for arg in "$@"; do
    printf -v arg '%q' "$arg"
    parts+=("$arg")
  done

  local joined=""
  if [[ ${#parts[@]} -gt 0 ]]; then
    printf -v joined '%s ' "${parts[@]}"
    joined="${joined% }"
  fi
  printf '%s\n' "$joined"
}

gpu_count() {
  local gpu_csv="$1"
  local -a parsed=()

  IFS=',' read -r -a parsed <<< "$gpu_csv"
  local count=0
  local gpu
  for gpu in "${parsed[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    if [[ -n "$gpu" ]]; then
      count=$((count + 1))
    fi
  done
  printf '%s\n' "$count"
}

resolve_config_file() {
  local model="$1"
  case "${model,,}" in
    apan)  printf '%s\n' "$SCRIPT_DIR/config/APAN.yml" ;;
    jodie) printf '%s\n' "$SCRIPT_DIR/config/JODIE.yml" ;;
    tgn)   printf '%s\n' "$SCRIPT_DIR/config/TGN.yml" ;;
    dyrep) printf '%s\n' "$SCRIPT_DIR/config/DyREP.yml" ;;
    *)
      echo "Unknown model: ${model}. Supported models: apan, jodie, tgn, dyrep" >&2
      exit 1
      ;;
  esac
}

parse_test_ap() {
  local log_path="$1"
  uv run python - "$log_path" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
matches = re.findall(
    r"^\s*test ap:([0-9.]+)\s+test auc:[0-9.]+",
    text,
    re.IGNORECASE | re.MULTILINE,
)
if not matches:
    raise SystemExit(f"Could not parse final test ap from {sys.argv[1]}")
print(matches[-1])
PY
}

summarize_run_dir() {
  local run_dir="$1"
  uv run python "$SCRIPT_DIR/scripts/summarize_seed_sweep.py" "$run_dir"
}

EMIT_JOBS=0
RUN_SINGLE_SEED=""
SUMMARIZE_ONLY=0
CLI_GPU_IDS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --emit-jobs)
      EMIT_JOBS=1
      shift
      ;;
    --run-seed)
      if [[ $# -lt 2 ]]; then
        echo "--run-seed requires a value" >&2
        exit 1
      fi
      RUN_SINGLE_SEED="$2"
      shift 2
      ;;
    --summarize-only)
      SUMMARIZE_ONLY=1
      shift
      ;;
    --gpus)
      if [[ $# -lt 2 ]]; then
        echo "--gpus requires a value" >&2
        exit 1
      fi
      CLI_GPU_IDS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "$SCRIPT_DIR"

GPU_IDS_DEFAULT="${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_IDS="${CLI_GPU_IDS:-${GPU_IDS:-$GPU_IDS_DEFAULT}}"
IFS=' ' read -r -a MODELS <<< "${MODELS:-apan}"
IFS=' ' read -r -a DATASETS <<< "${DATASETS:-LASTFM MOOC REDDIT WIKIPEDIA}"
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"
MACRO_BATCH_SIZE="${MACRO_BATCH_SIZE:-}"
EPOCHS="${EPOCHS:-100}"
STABLE_MODE="${STABLE_MODE:-true}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/seed_sweeps}"

case "${STABLE_MODE,,}" in
  1|true|yes|on)
    STABLE_MODE_ARG=true
    ;;
  *)
    echo "run_tgl_grid.sh only supports STABLE_MODE=true." >&2
    exit 1
    ;;
esac

if [[ "${#MODELS[@]}" -eq 0 || "${#DATASETS[@]}" -eq 0 || "${#SEEDS[@]}" -eq 0 ]]; then
  echo "MODELS, DATASETS, and SEEDS must all be non-empty." >&2
  exit 1
fi

run_dir_for() {
  local model="$1"
  local dataset="$2"
  local macro_batch_size

  macro_batch_size="$(macro_batch_size_for "$dataset")"
  printf '%s\n' "$RUN_ROOT/tgl_${model,,}_${dataset,,}_bs${macro_batch_size}"
}

macro_batch_size_for() {
  local dataset="$1"
  if [[ -n "$MACRO_BATCH_SIZE" ]]; then
    printf '%s\n' "$MACRO_BATCH_SIZE"
  else
    default_macro_batch_size "$dataset"
  fi
}

run_single_seed() {
  local seed="$1"
  local model="${MODEL:-}"
  local dataset="${DATASET:-}"
  local run_dir="${RUN_DIR:-}"
  local config_file=""
  local log_path=""
  local tmp_config=""
  local n_gpu=""
  local macro_batch_size=""
  local batch_size=""
  local test_ap=""
  local -a applied_stable_env=()

  if [[ -z "$model" || -z "$dataset" ]]; then
    echo "MODEL and DATASET must be set when using --run-seed." >&2
    exit 1
  fi

  if [[ -z "$run_dir" ]]; then
    run_dir="$(run_dir_for "$model" "$dataset")"
  fi
  mkdir -p "$run_dir"

  n_gpu="$(gpu_count "$GPU_IDS")"
  if (( n_gpu != 2 )); then
    echo "Each TGL job expects exactly 2 GPUs, got GPU_IDS=${GPU_IDS}" >&2
    exit 1
  fi
  macro_batch_size="$(macro_batch_size_for "$dataset")"
  if (( macro_batch_size % n_gpu != 0 )); then
    echo "MACRO_BATCH_SIZE (${macro_batch_size}) must be divisible by n_gpu (${n_gpu})." >&2
    exit 1
  fi
  batch_size=$((macro_batch_size / n_gpu))

  config_file="$(resolve_config_file "$model")"
  log_path="$run_dir/seed_${seed}.log"
  tmp_config="$(mktemp "$run_dir/${model^^}_XXXXXX.yml")"
  trap 'rm -f "$tmp_config"' EXIT

  sed \
    -e "0,/epoch: 10/s//epoch: ${EPOCHS}/" \
    -e "s/batch_size: [0-9]*/batch_size: ${batch_size}/" \
    "$config_file" > "$tmp_config"

  if [[ "$STABLE_MODE_ARG" == "true" ]]; then
    if [[ -z "${CUDA_DEVICE_MAX_CONNECTIONS+x}" ]]; then
      export CUDA_DEVICE_MAX_CONNECTIONS=1
      applied_stable_env+=("CUDA_DEVICE_MAX_CONNECTIONS=1")
    fi
    if [[ -z "${CUBLAS_WORKSPACE_CONFIG+x}" ]]; then
      export CUBLAS_WORKSPACE_CONFIG=:4096:8
      applied_stable_env+=("CUBLAS_WORKSPACE_CONFIG=:4096:8")
    fi
    stable_mode_var='FR''OST_STABLE_MODE'
    export "${stable_mode_var}"=1
  fi

  echo
  echo "=== [TGL ${model^^} 2GPU] dataset=${dataset} seed=${seed} gpus=${GPU_IDS} ==="
  echo "Epochs: ${EPOCHS}"
  echo "Macro batch size: ${macro_batch_size} (${batch_size} per GPU)"
  echo "Stable mode: ${STABLE_MODE_ARG}"
  if [[ ${#applied_stable_env[@]} -gt 0 ]]; then
    echo "Stable mode env overrides: ${applied_stable_env[*]}"
  fi
  echo "Temp config: ${tmp_config}"

  if ! CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    uv run python train_dist.py \
      --dataset "${dataset}" \
      --config "$tmp_config" \
      --num_gpus "${n_gpu}" \
      --seed "${seed}" \
      --rnd_edim 0 \
      --rnd_ndim 0 \
      --tqdm \
      2>&1 | tee "$log_path"; then
    echo "Run failed for dataset=${dataset} seed=${seed}. See ${log_path}" >&2
    exit 1
  fi

  test_ap="$(parse_test_ap "$log_path")"
  echo "[TGL ${model^^} 2GPU] dataset=${dataset} seed=${seed} final_test_ap=${test_ap}"
}

emit_jobs() {
  local model
  local dataset
  local seed
  local run_dir
  local log_path
  local desc
  local cmd

  mkdir -p "$RUN_ROOT"

  for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      run_dir="$(run_dir_for "$model" "$dataset")"
      mkdir -p "$run_dir"
      for seed in "${SEEDS[@]}"; do
        log_path="$run_dir/seed_${seed}.log"
        desc="TGL ${model^^} ${dataset} seed=${seed}"
        cmd="$(shell_join \
          env \
          "GPU_IDS=" \
          "MODEL=${model}" \
          "DATASET=${dataset}" \
          "MACRO_BATCH_SIZE=${MACRO_BATCH_SIZE}" \
          "EPOCHS=${EPOCHS}" \
          "STABLE_MODE=${STABLE_MODE_ARG}" \
          "RUN_DIR=${run_dir}" \
          bash \
          "$SCRIPT_PATH" \
          --run-seed \
          "$seed")"
        printf '%s\t%s\t%s\n' "$desc" "$log_path" "$cmd"
      done
    done
  done
}

summarize_all() {
  local model
  local dataset
  local run_dir
  local results_tsv
  local summary_tsv
  local summary_txt
  local missing=0

  for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      run_dir="$(run_dir_for "$model" "$dataset")"
      results_tsv="$run_dir/results.tsv"
      summary_tsv="$run_dir/summary.tsv"
      summary_txt="$run_dir/summary.txt"
      if [[ ! -d "$run_dir" ]]; then
        echo "Skipping missing run dir: $run_dir"
        missing=$((missing + 1))
        continue
      fi
      if ! compgen -G "$run_dir/seed_*.log" > /dev/null; then
        echo "Skipping run dir without seed logs: $run_dir"
        missing=$((missing + 1))
        continue
      fi

      echo
      echo "=== Summarizing TGL ${model^^} ${dataset} ==="
      summarize_run_dir "$run_dir" "$results_tsv" "$summary_txt"
      echo "Results TSV: $results_tsv"
      echo "Summary TSV: $summary_tsv"
      echo "Summary: $summary_txt"
    done
  done

  if (( missing > 0 )); then
    echo
    echo "Skipped ${missing} run directories during summary."
  fi
}

run_scheduler() {
  local -a runner_cmd=(
    uv run python "$PAIR_RUNNER"
    --script "$SCRIPT_PATH"
  )
  local num_gpus=""
  local num_pairs=""

  if [[ -n "$GPU_IDS" ]]; then
    num_gpus="$(gpu_count "$GPU_IDS")"
    if (( num_gpus == 0 )); then
      echo "No GPUs specified in GPU_IDS=${GPU_IDS}" >&2
      exit 1
    fi
    if (( num_gpus % 2 != 0 )); then
      echo "GPU_IDS must contain an even number of GPU ids, got: ${GPU_IDS}" >&2
      exit 1
    fi
    num_pairs=$((num_gpus / 2))
    runner_cmd+=(--gpus "$GPU_IDS")
  fi

  echo "Running TGL grid sweep with fine-grained GPU-pair scheduling"
  echo "Models: ${MODELS[*]}"
  echo "Datasets: ${DATASETS[*]}"
  echo "Seeds: ${SEEDS[*]}"
  if [[ -n "$MACRO_BATCH_SIZE" ]]; then
    echo "Macro batch size: ${MACRO_BATCH_SIZE}"
  else
    echo "Macro batch size: dataset defaults from ${DATASET_DEFAULTS_SH}"
  fi
  echo "Epochs: ${EPOCHS}"
  echo "Stable mode: ${STABLE_MODE_ARG}"
  echo "Run root: ${RUN_ROOT}"
  if [[ -n "$GPU_IDS" ]]; then
    echo "GPU pairs source: ${GPU_IDS}"
  else
    echo "GPU pairs source: auto-detect"
  fi

  set +e
  if [[ -n "$num_pairs" ]]; then
    MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-$num_pairs}" \
    MAX_JOBS_PER_GPU_PAIR="${MAX_JOBS_PER_GPU_PAIR:-1}" \
    PAIR_COOLDOWN_SECS="${PAIR_COOLDOWN_SECS:-0}" \
    "${runner_cmd[@]}"
  else
    MAX_JOBS_PER_GPU_PAIR="${MAX_JOBS_PER_GPU_PAIR:-1}" \
    PAIR_COOLDOWN_SECS="${PAIR_COOLDOWN_SECS:-0}" \
    "${runner_cmd[@]}"
  fi
  local scheduler_rc=$?
  set -e

  summarize_all
  return "$scheduler_rc"
}

if [[ "$SUMMARIZE_ONLY" -eq 1 ]]; then
  summarize_all
  exit 0
fi

if [[ "$EMIT_JOBS" -eq 1 ]]; then
  emit_jobs
  exit 0
fi

if [[ -n "$RUN_SINGLE_SEED" ]]; then
  run_single_seed "$RUN_SINGLE_SEED"
  exit 0
fi

run_scheduler
