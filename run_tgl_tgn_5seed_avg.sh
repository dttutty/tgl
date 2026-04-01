#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU_ID="${GPU_ID:-1}"
DATASET="${DATASET:-LASTFM}"
EPOCHS="${EPOCHS:-100}"
IFS=' ' read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/seed_sweeps/tgl_tgn_${DATASET,,}_${STAMP}}"
mkdir -p "$RUN_DIR"

RESULTS_TSV="$RUN_DIR/results.tsv"
SUMMARY_TXT="$RUN_DIR/summary.txt"
TMP_CONFIG="$(mktemp "$RUN_DIR/TGN_XXXXXX.yml")"

cleanup() {
  rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

sed "0,/epoch: 10/s//epoch: ${EPOCHS}/" config/TGN.yml > "$TMP_CONFIG"

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

echo "Running TGL TGN seed sweep on GPU ${GPU_ID}"
echo "Dataset: ${DATASET}"
echo "Epochs: ${EPOCHS}"
echo "Seeds: ${SEEDS[*]}"
echo "Temp config: ${TMP_CONFIG}"
echo "Run directory: ${RUN_DIR}"

for seed in "${SEEDS[@]}"; do
  log_path="$RUN_DIR/seed_${seed}.log"
  echo
  echo "=== [TGL] seed=${seed} ==="
  if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    uv run python train_dist.py \
      --dataset "${DATASET}" \
      --config "$TMP_CONFIG" \
      --num_gpus 1 \
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
  echo "[TGL] seed=${seed} final_test_ap=${test_ap}"
done

mean_test_ap "$RESULTS_TSV" | tee "$SUMMARY_TXT"
echo "Results TSV: $RESULTS_TSV"
echo "Summary: $SUMMARY_TXT"
