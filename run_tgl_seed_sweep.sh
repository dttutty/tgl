#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 通用参数
GPU_IDS="${GPU_IDS:-0,1}"
MODEL="${MODEL:-apan}"
STABLE_MODE="${STABLE_MODE:-true}"

# 全局默认值
DATASET="${DATASET:-LASTFM}"
EPOCHS="${EPOCHS:-100}"

# 模型特定配置
case "${MODEL,,}" in
  apan)
    CONFIG_FILE="config/APAN.yml"
    ;;
  jodie)
    CONFIG_FILE="config/JODIE.yml"
    ;;
  tgn)
    CONFIG_FILE="config/TGN.yml"
    ;;
  dyrep)
    CONFIG_FILE="config/DyREP.yml"
    ;;
  *)
    echo "Unknown model: ${MODEL}. Supported models: apan, jodie, tgn, dyrep" >&2
    exit 1
    ;;
esac

IFS=' ' read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"

REPO_ROOT="${SCRIPT_DIR}/../.."
source "${REPO_ROOT}/DATA/dataset_defaults.sh"

N_GPU="$(tr ',' '\n' <<< "${GPU_IDS}" | grep -c .)"
MACRO_BATCH_SIZE="${MACRO_BATCH_SIZE:-$(default_macro_batch_size "${DATASET}")}"
BATCH_SIZE="${BATCH_SIZE:-$(( MACRO_BATCH_SIZE / N_GPU ))}"

case "${STABLE_MODE,,}" in
  1|true|yes|on) STABLE_MODE_ARG=true ;;
  0|false|no|off) STABLE_MODE_ARG=false ;;
  *)
    echo "Invalid STABLE_MODE=${STABLE_MODE}. Use true/false." >&2
    exit 1
    ;;
esac

applied_stable_env=()
if [[ "$STABLE_MODE_ARG" == "true" ]]; then
  if [[ -z "${CUDA_DEVICE_MAX_CONNECTIONS+x}" ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    applied_stable_env+=("CUDA_DEVICE_MAX_CONNECTIONS=1")
  fi
  if [[ -z "${CUBLAS_WORKSPACE_CONFIG+x}" ]]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    applied_stable_env+=("CUBLAS_WORKSPACE_CONFIG=:4096:8")
  fi
  export FROST_STABLE_MODE=1
fi

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/seed_sweeps/tgl_${MODEL,,}_${DATASET,,}_2gpu_${STAMP}}"
mkdir -p "$RUN_DIR"

RESULTS_TSV="$RUN_DIR/results.tsv"
SUMMARY_TSV="$RUN_DIR/summary.tsv"
SUMMARY_TXT="$RUN_DIR/summary.txt"
TMP_CONFIG="$(mktemp "$RUN_DIR/${MODEL^^}_XXXXXX.yml")"

cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

sed \
  -e "0,/epoch: 10/s//epoch: ${EPOCHS}/" \
  -e "s/batch_size: [0-9]*/batch_size: ${BATCH_SIZE}/" \
  "${CONFIG_FILE}" > "$TMP_CONFIG"

printf "seed\ttest_ap\tlog_path\n" > "$RESULTS_TSV"

parse_test_ap() {
  local log_path="$1"
  uv run python - "$log_path" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
matches = re.findall(r"^\s*test ap:([0-9.]+)\s+test auc:[0-9.]+", text, re.IGNORECASE | re.MULTILINE)
if not matches:
    raise SystemExit(f"Could not parse final test ap from {sys.argv[1]}")
print(matches[-1])
PY
}

echo "Running TGL ${MODEL^^} seed sweep on GPUs ${GPU_IDS}"
echo "Dataset: ${DATASET}"
echo "Epochs: ${EPOCHS}"
echo "Macro batch size: ${MACRO_BATCH_SIZE} (${BATCH_SIZE} per GPU)"
echo "Stable mode: ${STABLE_MODE_ARG}"
if [[ ${#applied_stable_env[@]} -gt 0 ]]; then
  echo "Stable mode env overrides: ${applied_stable_env[*]}"
fi
echo "Seeds: ${SEEDS[*]}"
echo "Temp config: ${TMP_CONFIG}"
echo "Run directory: ${RUN_DIR}"

for seed in "${SEEDS[@]}"; do
  log_path="$RUN_DIR/seed_${seed}.log"
  echo
  echo "=== [TGL ${MODEL^^} 2GPU] seed=${seed} ==="
  if ! CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    uv run python train_dist.py \
      --dataset "${DATASET}" \
      --config "$TMP_CONFIG" \
      --num_gpus "${N_GPU}" \
      --seed "${seed}" \
      --rnd_edim 0 \
      --rnd_ndim 0 \
      --tqdm \
      2>&1 | tee "$log_path"; then
    echo "Run failed for seed=${seed}. See ${log_path}" >&2
    exit 1
  fi

  test_ap="$(parse_test_ap "$log_path")"
  printf "%s\t%s\t%s\n" "$seed" "$test_ap" "$log_path" >> "$RESULTS_TSV"
  echo "[TGL ${MODEL^^} 2GPU] seed=${seed} final_test_ap=${test_ap}"
done

uv run python "$SCRIPT_DIR/scripts/summarize_seed_sweep.py" "$RUN_DIR"
echo "Results TSV: $RESULTS_TSV"
echo "Summary TSV: $SUMMARY_TSV"
echo "Summary: $SUMMARY_TXT"
