#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
MODELS_RAW="${MODELS:-apan jodie tgn}"
DATASETS_RAW="${DATASETS:-LASTFM MOOC REDDIT WIKIPEDIA}"
BATCH_SIZES_RAW="${BATCH_SIZES:-600 1200 2000 4000}"
SEEDS_RAW="${SEEDS:-0}"
WARMUP_BATCHES="${WARMUP_BATCHES:-5}"
MEASURE_BATCHES="${MEASURE_BATCHES:-100}"
REQUIRE_FULL_MEASURE="${REQUIRE_FULL_MEASURE:-1}"
TGL_EPOCHS="${TGL_EPOCHS:-1000}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-$(( WARMUP_BATCHES + MEASURE_BATCHES + 1 ))}"

if (( WARMUP_BATCHES < 0 )); then
  echo "WARMUP_BATCHES must be >= 0, got ${WARMUP_BATCHES}" >&2
  exit 1
fi
if (( MEASURE_BATCHES < 1 )); then
  echo "MEASURE_BATCHES must be >= 1, got ${MEASURE_BATCHES}" >&2
  exit 1
fi
if (( MAX_TRAIN_STEPS < 0 )); then
  echo "MAX_TRAIN_STEPS must be >= 0, got ${MAX_TRAIN_STEPS}" >&2
  exit 1
fi

MODELS_NORM="${MODELS_RAW//,/ }"
DATASETS_NORM="${DATASETS_RAW//,/ }"
BATCH_SIZES_NORM="${BATCH_SIZES_RAW//,/ }"
SEEDS_NORM="${SEEDS_RAW//,/ }"
IFS=' ' read -r -a MODELS <<< "${MODELS_NORM}"
IFS=' ' read -r -a DATASETS <<< "${DATASETS_NORM}"
IFS=' ' read -r -a BATCH_SIZES <<< "${BATCH_SIZES_NORM}"
IFS=' ' read -r -a SEEDS <<< "${SEEDS_NORM}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
BASE_DIR="${REPO_ROOT}/experiments/com_throughput_with_tgl_serial_train"
RUN_DIR="${RUN_DIR:-${BASE_DIR}/throughput_serial_pinmem_${STAMP}}"
LOG_DIR="${RUN_DIR}/logs"
CFG_DIR="${RUN_DIR}/configs"
mkdir -p "${LOG_DIR}" "${CFG_DIR}"

RESULTS_TSV="${RUN_DIR}/results.tsv"
SUMMARY_TSV="${RUN_DIR}/summary.tsv"
SUMMARY_TXT="${RUN_DIR}/summary.txt"
PROGRESS_MD="${RUN_DIR}/progress.md"

base_tgl_config_for_model() {
  local model="${1,,}"
  case "${model}" in
    tgn) echo "${SCRIPT_DIR}/config/TGN.yml" ;;
    jodie) echo "${SCRIPT_DIR}/config/JODIE.yml" ;;
    apan) echo "${SCRIPT_DIR}/config/APAN.yml" ;;
    *)
      echo "Unsupported model=${1}. Supported: apan, jodie, tgn" >&2
      return 1
      ;;
  esac
}

frost_summary_for_model() {
  local model="${1,,}"
  case "${model}" in
    tgn)
      echo "${REPO_ROOT}/experiments/com_throughput_with_tgl/throughput_2gpu_fair_tgn_first_20260414_092725/summary.tsv"
      ;;
    jodie)
      echo "${REPO_ROOT}/experiments/com_throughput_with_tgl/throughput_2gpu_fair_jodie_20260414_100014/summary.tsv"
      ;;
    apan)
      echo "${REPO_ROOT}/experiments/com_throughput_with_tgl/throughput_2gpu_fair_apan_after_tgn_20260414_094002/summary.tsv"
      ;;
    *)
      echo "Unsupported model=${1}. Supported: apan, jodie, tgn" >&2
      return 1
      ;;
  esac
}

write_tgl_config() {
  local src_cfg="$1"
  local dst_cfg="$2"
  local epoch="$3"
  local batch_size="$4"
  sed -E \
    -e "0,/epoch:[[:space:]]*[0-9]+/s//epoch: ${epoch}/" \
    -e "0,/batch_size:[[:space:]]*[0-9]+/s//batch_size: ${batch_size}/" \
    "${src_cfg}" > "${dst_cfg}"
}

lookup_frost_summary_row() {
  local summary_tsv="$1"
  local model="$2"
  local dataset="$3"
  local macro_batch_size="$4"
  uv run python - "$summary_tsv" "$model" "$dataset" "$macro_batch_size" <<'PY'
import csv
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
model = sys.argv[2].lower()
dataset = sys.argv[3].upper()
macro_batch_size = sys.argv[4]
with summary_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if (
            row["model"].lower() == model
            and row["dataset"].upper() == dataset
            and row["macro_batch_size"] == macro_batch_size
        ):
            print(
                "\t".join(
                    [
                        row["frost_events_per_sec_per_gpu_mean"],
                        str(row.get("runs", "1")),
                        row.get("speedup_frost_over_tgl_mean", ""),
                    ]
                )
            )
            raise SystemExit(0)
raise SystemExit(
    f"Could not find FROST summary row for model={model} dataset={dataset} macro_batch_size={macro_batch_size} in {summary_path}"
)
PY
}

parse_tgl_serial_events_per_sec_per_gpu() {
  local log_path="$1"
  uv run python - "$log_path" <<'PY'
import json
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
decoder = json.JSONDecoder()
rows = []
for match in re.finditer(r"APPENDIX_TRAIN_E2E_SUMMARY\s+", text):
    start = match.end()
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        continue
    v = obj.get("events_per_sec_per_gpu")
    if not isinstance(v, (int, float)):
        continue
    interval_count = obj.get("interval_count")
    rows.append(
        (
            float(v),
            int(interval_count) if isinstance(interval_count, int) else -1,
        )
    )
if not rows:
    raise SystemExit(f"Could not parse serial TGL throughput from {sys.argv[1]}")
eps, interval_count = rows[-1]
print(f"{eps:.6f}\t{interval_count}")
PY
}

printf "model\tdataset\tseed\tgpu_id\tfrost_macro_batch_size\tfrost_per_gpu_batch_size\ttgl_serial_batch_size\tfrost_events_per_sec_per_gpu\ttgl_serial_events_per_sec_per_gpu\tspeedup_frost_over_tgl_serial\tfrost_reference_runs\ttgl_serial_measured_batches\tfrost_summary_path\ttgl_log\n" > "${RESULTS_TSV}"

cat > "${PROGRESS_MD}" <<EOF
# Serial TGL Train Throughput Sweep

- Run dir: ${RUN_DIR}
- GPU_ID: ${GPU_ID}
- Models: ${MODELS[*]}
- Datasets: ${DATASETS[*]}
- Batch sizes: ${BATCH_SIZES[*]}
- Seeds: ${SEEDS[*]}
- TGL script: \`third_party/tgl/train.py --pin_memory\`
- Measurement: skip ${WARMUP_BATCHES} warm-up train intervals, measure next ${MEASURE_BATCHES} train intervals
- Max train steps per run: ${MAX_TRAIN_STEPS}
- Memory update delay: disabled

## Progress

EOF

echo "=== Serial TGL train.py Throughput Benchmark ==="
echo "Run dir: ${RUN_DIR}"
echo "GPU_ID: ${GPU_ID}"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Warmup batches: ${WARMUP_BATCHES}"
echo "Measure batches: ${MEASURE_BATCHES}"
echo "Max train steps: ${MAX_TRAIN_STEPS}"
echo "Require full measurement window: ${REQUIRE_FULL_MEASURE}"

combo_idx=0
for model in "${MODELS[@]}"; do
  model_lc="${model,,}"
  model_uc="${model_lc^^}"
  src_cfg="$(base_tgl_config_for_model "${model_lc}")"
  frost_summary="$(frost_summary_for_model "${model_lc}")"
  for dataset in "${DATASETS[@]}"; do
    dataset_uc="${dataset^^}"
    dataset_lc="${dataset,,}"
    for batch_size in "${BATCH_SIZES[@]}"; do
      cfg_path="${CFG_DIR}/${model_lc}_${dataset_lc}_bs${batch_size}.yml"
      write_tgl_config "${src_cfg}" "${cfg_path}" "${TGL_EPOCHS}" "${batch_size}"
      frost_per_gpu_batch_size="$(( batch_size / 2 ))"
      for seed in "${SEEDS[@]}"; do
        combo_idx=$(( combo_idx + 1 ))
        log_path="${LOG_DIR}/tgl_serial_${model_lc}_${dataset_lc}_bs${batch_size}_seed${seed}.log"

        echo
        echo "=== ${model_uc} ${dataset_uc} batch=${batch_size} seed=${seed} ==="
        echo "Using GPU ${GPU_ID}; this run is strictly single-instance."

        if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" \
          uv run --project "${SCRIPT_DIR}" python "${SCRIPT_DIR}/train.py" \
            --data "${dataset_uc}" \
            --config "${cfg_path}" \
            --gpu 0 \
            --seed "${seed}" \
            --batch_size "${batch_size}" \
            --pin_memory \
            --appendix-train-e2e \
            --measure-start-epoch 0 \
            --warmup-batches "${WARMUP_BATCHES}" \
            --measure-batches "${MEASURE_BATCHES}" \
            --max-train-steps "${MAX_TRAIN_STEPS}" \
            > "${log_path}" 2>&1; then
          echo "Serial TGL run failed. See ${log_path}" >&2
          echo "- FAILED: ${model_lc}/${dataset_uc}/bs${batch_size}/seed${seed} -> ${log_path}" >> "${PROGRESS_MD}"
          exit 1
        fi

        read -r tgl_eps tgl_measured_batches < <(parse_tgl_serial_events_per_sec_per_gpu "${log_path}")
        echo "[TGL serial] events/sec/gpu=${tgl_eps} measured_batches=${tgl_measured_batches}"
        if (( tgl_measured_batches >= 0 && tgl_measured_batches < MEASURE_BATCHES )); then
          if (( REQUIRE_FULL_MEASURE == 1 )); then
            echo "Serial TGL measured only ${tgl_measured_batches} batches (< ${MEASURE_BATCHES}) for ${model_lc}/${dataset_uc}/bs${batch_size}" >&2
            echo "- FAILED_WINDOW: ${model_lc}/${dataset_uc}/bs${batch_size}/seed${seed} measured ${tgl_measured_batches}" >> "${PROGRESS_MD}"
            exit 1
          fi
          echo "Warning: Serial TGL measured only ${tgl_measured_batches} batches (< ${MEASURE_BATCHES}) for ${model_lc}/${dataset_uc}/bs${batch_size}" >&2
        fi

        read -r frost_eps frost_runs _old_speedup < <(lookup_frost_summary_row "${frost_summary}" "${model_lc}" "${dataset_uc}" "${batch_size}")
        speedup="$(uv run python - "${frost_eps}" "${tgl_eps}" <<'PY'
import sys
f = float(sys.argv[1])
t = float(sys.argv[2])
print(f"{(f / t) if t > 0 else float('nan'):.6f}")
PY
)"

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${model_lc}" "${dataset_uc}" "${seed}" "${GPU_ID}" \
          "${batch_size}" "${frost_per_gpu_batch_size}" "${batch_size}" \
          "${frost_eps}" "${tgl_eps}" "${speedup}" \
          "${frost_runs}" "${tgl_measured_batches}" \
          "${frost_summary}" "${log_path}" \
          >> "${RESULTS_TSV}"

        echo "[SUMMARY] FROST/TGL-serial speedup=${speedup}x"
        echo "- DONE: ${model_lc}/${dataset_uc}/bs${batch_size}/seed${seed} -> speedup ${speedup}x" >> "${PROGRESS_MD}"
      done
    done
  done
done

uv run python - "${RESULTS_TSV}" "${SUMMARY_TSV}" "${SUMMARY_TXT}" <<'PY'
import csv
import pathlib
import statistics
import sys

results_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
summary_txt_path = pathlib.Path(sys.argv[3])

with results_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)

grouped = {}
for row in rows:
    key = (row["model"], row["dataset"], row["frost_macro_batch_size"])
    grouped.setdefault(key, []).append(row)

summary_rows = []
for key in sorted(grouped):
    items = grouped[key]
    frost = statistics.mean(float(x["frost_events_per_sec_per_gpu"]) for x in items)
    tgl = statistics.mean(float(x["tgl_serial_events_per_sec_per_gpu"]) for x in items)
    speedup = statistics.mean(float(x["speedup_frost_over_tgl_serial"]) for x in items)
    summary_rows.append(
        {
            "model": key[0],
            "dataset": key[1],
            "batch_size": key[2],
            "runs": str(len(items)),
            "frost_events_per_sec_per_gpu_mean": f"{frost:.6f}",
            "tgl_serial_events_per_sec_per_gpu_mean": f"{tgl:.6f}",
            "speedup_frost_over_tgl_serial_mean": f"{speedup:.6f}",
        }
    )

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "dataset",
            "batch_size",
            "runs",
            "frost_events_per_sec_per_gpu_mean",
            "tgl_serial_events_per_sec_per_gpu_mean",
            "speedup_frost_over_tgl_serial_mean",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(summary_rows)

lines = [
    "Serial TGL train.py vs existing FROST throughput summary",
    f"Source results: {results_path}",
    "",
]
for r in summary_rows:
    lines.append(
        f"{r['model']} {r['dataset']} batch={r['batch_size']}: "
        f"FROST {r['frost_events_per_sec_per_gpu_mean']} ev/s/gpu, "
        f"TGL-serial {r['tgl_serial_events_per_sec_per_gpu_mean']} ev/s/gpu, "
        f"speedup {r['speedup_frost_over_tgl_serial_mean']}x"
    )
summary_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo
echo "Finished. Results:"
echo "  ${RESULTS_TSV}"
echo "  ${SUMMARY_TSV}"
echo "  ${SUMMARY_TXT}"
echo "  ${PROGRESS_MD}"
