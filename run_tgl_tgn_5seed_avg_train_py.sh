#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU_ID="${GPU_ID:-1}"
DATASET="${DATASET:-LASTFM}"
EPOCHS="${EPOCHS:-100}"
SAMPLER_THREADS="${SAMPLER_THREADS:-1}"

REPO_ROOT="${SCRIPT_DIR}/../.."
# shellcheck source=../../DATA/dataset_defaults.sh
source "${REPO_ROOT}/DATA/dataset_defaults.sh"

MACRO_BATCH_SIZE="${MACRO_BATCH_SIZE:-$(default_macro_batch_size "${DATASET}")}"
BATCH_SIZE="${BATCH_SIZE:-${MACRO_BATCH_SIZE}}"
STABLE_MODE="${STABLE_MODE:-true}"
PIN_MEMORY="${PIN_MEMORY:-false}"
MEMORY_UPDATE_DELAY_BATCHES="${MEMORY_UPDATE_DELAY_BATCHES:-0}"
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"

case "${STABLE_MODE,,}" in
  1|true|yes|on) STABLE_MODE_ARG=true ;;
  0|false|no|off) STABLE_MODE_ARG=false ;;
  *)
    echo "Invalid STABLE_MODE=${STABLE_MODE}. Use true/false." >&2
    exit 1
    ;;
esac

case "${PIN_MEMORY,,}" in
  1|true|yes|on) PIN_MEMORY_ARG=true ;;
  0|false|no|off) PIN_MEMORY_ARG=false ;;
  *)
    echo "Invalid PIN_MEMORY=${PIN_MEMORY}. Use true/false." >&2
    exit 1
    ;;
esac

if ! [[ "$SAMPLER_THREADS" =~ ^[0-9]+$ ]] || [[ "$SAMPLER_THREADS" -lt 1 ]]; then
  echo "Invalid SAMPLER_THREADS=${SAMPLER_THREADS}. Use an integer >= 1." >&2
  exit 1
fi

if ! [[ "$MEMORY_UPDATE_DELAY_BATCHES" =~ ^[0-9]+$ ]]; then
  echo "Invalid MEMORY_UPDATE_DELAY_BATCHES=${MEMORY_UPDATE_DELAY_BATCHES}. Use an integer >= 0." >&2
  exit 1
fi

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
  if [[ -z "${OMP_NUM_THREADS+x}" ]]; then
    export OMP_NUM_THREADS=1
    applied_stable_env+=("OMP_NUM_THREADS=1")
  fi
  if [[ -z "${MKL_NUM_THREADS+x}" ]]; then
    export MKL_NUM_THREADS=1
    applied_stable_env+=("MKL_NUM_THREADS=1")
  fi
  export FROST_STABLE_MODE=1
fi

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/seed_sweeps/tgl_tgn_train_py_${DATASET,,}_${STAMP}}"
mkdir -p "$RUN_DIR"

RESULTS_TSV="$RUN_DIR/results.tsv"
SUMMARY_TXT="$RUN_DIR/summary.txt"
TMP_CONFIG="$(mktemp "$RUN_DIR/TGN_XXXXXX.yml")"

cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

sed \
  -e "0,/epoch: 10/s//epoch: ${EPOCHS}/" \
  -e "s/batch_size: [0-9]*/batch_size: ${BATCH_SIZE}/" \
  -e "0,/num_thread: 32/s//num_thread: ${SAMPLER_THREADS}/" \
  config/TGN.yml > "$TMP_CONFIG"

printf "seed\ttest_ap\tlog_path\n" > "$RESULTS_TSV"

parse_test_ap() {
  local log_path="$1"
  uv run python - "$log_path" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
matches = re.findall(r"^\s*test ap:([0-9.]+)\s+test (?:auc|mrr):[0-9.]+", text, re.IGNORECASE | re.MULTILINE)
if not matches:
    raise SystemExit(f"Could not parse final test ap from {sys.argv[1]}")
print(matches[-1])
PY
}

mean_test_ap() {
  local results_tsv="$1"
  uv run python - "$results_tsv" <<'PY'
import csv
import statistics
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

values = [float(row["test_ap"]) for row in rows]
mean = sum(values) / len(values)
stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
print(f"mean_test_ap={mean:.6f}")
print(f"std_test_ap={stdev:.6f}")
PY
}

echo "Running TGL TGN seed sweep via train.py on GPU ${GPU_ID}"
echo "Dataset: ${DATASET}"
echo "Epochs: ${EPOCHS}"
echo "Macro batch size: ${MACRO_BATCH_SIZE} (${BATCH_SIZE} per GPU)"
echo "Sampler threads: ${SAMPLER_THREADS}"
echo "Pin memory: ${PIN_MEMORY_ARG}"
echo "Memory update delay batches: ${MEMORY_UPDATE_DELAY_BATCHES}"
echo "Stable mode: ${STABLE_MODE_ARG}"
if [[ ${#applied_stable_env[@]} -gt 0 ]]; then
  echo "Stable mode env overrides: ${applied_stable_env[*]}"
fi
echo "Seeds: ${SEEDS[*]}"
echo "Temp config: ${TMP_CONFIG}"
echo "Run directory: ${RUN_DIR}"

for seed in "${SEEDS[@]}"; do
  log_path="$RUN_DIR/seed_${seed}.log"
  cmd=(
    env PYTHONUNBUFFERED=1 uv run python -u train.py
    --data "${DATASET}"
    --config "$TMP_CONFIG"
    --gpu "${GPU_ID}"
    --seed "${seed}"
    --rand_edge_features 0
    --rand_node_features 0
    --memory_update_delay_batches "${MEMORY_UPDATE_DELAY_BATCHES}"
  )
  if [[ "$PIN_MEMORY_ARG" == "true" ]]; then
    cmd+=(--pin_memory)
  fi

  echo
  echo "=== [TGL TGN train.py] seed=${seed} ==="
  if ! "${cmd[@]}" 2>&1 | tee "$log_path"; then
    echo "Run failed for seed=${seed}. See ${log_path}" >&2
    exit 1
  fi

  test_ap="$(parse_test_ap "$log_path")"
  printf "%s\t%s\t%s\n" "$seed" "$test_ap" "$log_path" >> "$RESULTS_TSV"
  echo "[TGL TGN train.py] seed=${seed} final_test_ap=${test_ap}"
done

mean_test_ap "$RESULTS_TSV" | tee "$SUMMARY_TXT"
echo "Results TSV: $RESULTS_TSV"
echo "Summary: $SUMMARY_TXT"
