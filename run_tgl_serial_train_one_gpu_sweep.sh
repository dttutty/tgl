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
TGL_PIN_MEMORY="${TGL_PIN_MEMORY:-1}"
WARMUP_BATCHES="${WARMUP_BATCHES:-5}"
MEASURE_BATCHES="${MEASURE_BATCHES:-100}"
REQUIRE_FULL_MEASURE="${REQUIRE_FULL_MEASURE:-1}"
TGL_EPOCHS="${TGL_EPOCHS:-1000}"
TGL_MAX_TRAIN_STEPS_OVERRIDE="${TGL_MAX_TRAIN_STEPS:-}"

if (( WARMUP_BATCHES < 0 )); then
  echo "WARMUP_BATCHES must be >= 0, got ${WARMUP_BATCHES}" >&2
  exit 1
fi
if [[ "${TGL_PIN_MEMORY}" != "0" && "${TGL_PIN_MEMORY}" != "1" ]]; then
  echo "TGL_PIN_MEMORY must be 0 or 1, got ${TGL_PIN_MEMORY}" >&2
  exit 1
fi
if (( MEASURE_BATCHES < 1 )); then
  echo "MEASURE_BATCHES must be >= 1, got ${MEASURE_BATCHES}" >&2
  exit 1
fi
if [[ -n "${TGL_MAX_TRAIN_STEPS_OVERRIDE}" ]] && (( TGL_MAX_TRAIN_STEPS_OVERRIDE < 0 )); then
  echo "TGL_MAX_TRAIN_STEPS must be >= 0, got ${TGL_MAX_TRAIN_STEPS_OVERRIDE}" >&2
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
BASE_DIR="${REPO_ROOT}/experiments/com_throughput_with_tgl_serial_train_one_gpu"
if [[ "${TGL_PIN_MEMORY}" == "1" ]]; then
  RUN_BASENAME="throughput_1gpu_pinmem_${STAMP}"
else
  RUN_BASENAME="throughput_1gpu_nopinmem_${STAMP}"
fi
RUN_DIR="${RUN_DIR:-${BASE_DIR}/${RUN_BASENAME}}"
LOG_DIR="${RUN_DIR}/logs"
CFG_DIR="${RUN_DIR}/configs"
mkdir -p "${LOG_DIR}" "${CFG_DIR}"

RESULTS_TSV="${RUN_DIR}/results.tsv"
SUMMARY_TSV="${RUN_DIR}/summary.tsv"
SUMMARY_TXT="${RUN_DIR}/summary.txt"
REPORT_MD="${RUN_DIR}/report.md"
PROGRESS_MD="${RUN_DIR}/progress.md"

if [[ -z "${CUDA_DEVICE_MAX_CONNECTIONS+x}" ]]; then
  export CUDA_DEVICE_MAX_CONNECTIONS=1
fi
if [[ -z "${CUBLAS_WORKSPACE_CONFIG+x}" ]]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
export FROST_STABLE_MODE=1

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

compute_frost_epochs() {
  local dataset="$1"
  local batch_size="$2"
  local warmup="$3"
  local target="$4"
  uv run python - "$dataset" "$batch_size" "$warmup" "$target" <<'PY'
import math
import pandas as pd
import sys
from pathlib import Path

dataset = sys.argv[1].upper()
batch_size = int(sys.argv[2])
warmup = int(sys.argv[3])
target = int(sys.argv[4])
df = pd.read_csv(Path("DATA") / dataset / "edges.csv")
train_edge_end = int(df[df["default_split"].gt(0)].index[0])
steps_per_epoch = math.ceil(train_edge_end / batch_size)
measured_per_epoch = max(steps_per_epoch - warmup, 0)
if measured_per_epoch <= 0:
    raise SystemExit(
        f"Not enough train steps per epoch for dataset={dataset}, "
        f"batch_size={batch_size}, warmup={warmup}: {steps_per_epoch} steps/epoch"
    )
epochs = math.ceil(target / measured_per_epoch)
print(max(1, epochs))
PY
}

compute_tgl_max_train_steps() {
  local dataset="$1"
  local batch_size="$2"
  local warmup="$3"
  local target="$4"
  uv run python - "$dataset" "$batch_size" "$warmup" "$target" <<'PY'
import math
import pandas as pd
import sys
from pathlib import Path

dataset = sys.argv[1].upper()
batch_size = int(sys.argv[2])
warmup = int(sys.argv[3])
target = int(sys.argv[4])
df = pd.read_csv(Path("DATA") / dataset / "edges.csv")
train_edge_end = int(df[df["default_split"].gt(0)].index[0])
steps_per_epoch = math.ceil(train_edge_end / batch_size)
required_intervals = warmup + target
steps = 1
while True:
    intervals = steps - math.ceil(steps / steps_per_epoch)
    if intervals >= required_intervals:
        print(steps)
        raise SystemExit(0)
    steps += 1
PY
}

parse_frost_events_per_sec_per_gpu() {
  local log_path="$1"
  uv run python - "$log_path" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
matches = re.findall(
    r"TRAIN_THROUGHPUT_PER_GPU\s+rank=(\d+)\s+events_per_sec_per_gpu=([0-9eE+\-.]+).*?measured_batches=([0-9]+)",
    text,
)
if not matches:
    raise SystemExit(f"Could not parse FROST throughput from {sys.argv[1]}")
by_rank = {}
for rank_s, eps_s, measured_s in matches:
    by_rank[int(rank_s)] = (float(eps_s), int(measured_s))
eps_values = [value[0] for value in by_rank.values()]
measured_values = [value[1] for value in by_rank.values()]
print(f"{min(eps_values):.6f}\t{min(measured_values)}")
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

if [[ ! -s "${RESULTS_TSV}" ]]; then
  printf "model\tdataset\tseed\tgpu_id\tbatch_size\tfrost_epochs\ttgl_max_train_steps\tfrost_events_per_sec_per_gpu\ttgl_serial_events_per_sec_per_gpu\tspeedup_frost_over_tgl\tfrost_measured_batches\ttgl_serial_measured_batches\tfrost_log\ttgl_log\n" > "${RESULTS_TSV}"
fi

if [[ ! -s "${PROGRESS_MD}" ]]; then
cat > "${PROGRESS_MD}" <<EOF
# FROST 1GPU vs TGL train.py 1GPU

- Run dir: ${RUN_DIR}
- GPU_ID: ${GPU_ID}
- Models: ${MODELS[*]}
- Datasets: ${DATASETS[*]}
- Batch sizes: ${BATCH_SIZES[*]}
- Seeds: ${SEEDS[*]}
- TGL script: \`third_party/tgl/train.py$( [[ "${TGL_PIN_MEMORY}" == "1" ]] && printf ' --pin_memory' ) --memory_update_delay_batches 0\`
- FROST launch: \`scripts/train_preset.py\` with \`--nproc-per-node=2\`
- Measurement: FROST skips ${WARMUP_BATCHES} warmup batches per epoch and accumulates ${MEASURE_BATCHES} measured batches total; TGL serial skips the first ${WARMUP_BATCHES} train intervals and measures the next ${MEASURE_BATCHES}
- Stable env: \`CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}\`, \`CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}\`

## Progress

EOF
fi

result_already_recorded() {
  local model="$1"
  local dataset="$2"
  local seed="$3"
  local batch_size="$4"
  uv run python - "$RESULTS_TSV" "$model" "$dataset" "$seed" "$batch_size" <<'PY'
import csv
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists() or path.stat().st_size == 0:
    raise SystemExit(1)
model, dataset, seed, batch_size = sys.argv[2:6]
with path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if (
            row["model"] == model
            and row["dataset"] == dataset
            and row["seed"] == seed
            and row["batch_size"] == batch_size
        ):
            raise SystemExit(0)
raise SystemExit(1)
PY
}

echo "=== One-GPU Throughput Benchmark (FROST vs TGL train.py) ==="
echo "Run dir: ${RUN_DIR}"
echo "GPU_ID: ${GPU_ID}"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "TGL pin_memory: ${TGL_PIN_MEMORY}"
echo "Stable env: CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}, CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}, FROST_STABLE_MODE=${FROST_STABLE_MODE}"
echo "Warmup batches: ${WARMUP_BATCHES}"
echo "Measure batches: ${MEASURE_BATCHES}"
if [[ -n "${TGL_MAX_TRAIN_STEPS_OVERRIDE}" ]]; then
  echo "TGL max train steps: override=${TGL_MAX_TRAIN_STEPS_OVERRIDE}"
else
  echo "TGL max train steps: auto-computed per dataset/batch"
fi

combo_idx=0
for model in "${MODELS[@]}"; do
  model_lc="${model,,}"
  model_uc="${model_lc^^}"
  src_cfg="$(base_tgl_config_for_model "${model_lc}")"
  for dataset in "${DATASETS[@]}"; do
    dataset_uc="${dataset^^}"
    dataset_lc="${dataset,,}"
    for batch_size in "${BATCH_SIZES[@]}"; do
      cfg_path="${CFG_DIR}/${model_lc}_${dataset_lc}_bs${batch_size}.yml"
      write_tgl_config "${src_cfg}" "${cfg_path}" "${TGL_EPOCHS}" "${batch_size}"
      frost_epochs="$(compute_frost_epochs "${dataset_uc}" "${batch_size}" "${WARMUP_BATCHES}" "${MEASURE_BATCHES}")"
      if [[ -n "${TGL_MAX_TRAIN_STEPS_OVERRIDE}" ]]; then
        tgl_max_train_steps="${TGL_MAX_TRAIN_STEPS_OVERRIDE}"
      else
        tgl_max_train_steps="$(compute_tgl_max_train_steps "${dataset_uc}" "${batch_size}" "${WARMUP_BATCHES}" "${MEASURE_BATCHES}")"
      fi

      for seed in "${SEEDS[@]}"; do
        if result_already_recorded "${model_lc}" "${dataset_uc}" "${seed}" "${batch_size}"; then
          echo "Skipping completed combo ${model_lc}/${dataset_uc}/bs${batch_size}/seed${seed}"
          continue
        fi
        combo_idx=$(( combo_idx + 1 ))
        frost_log="${LOG_DIR}/frost_${model_lc}_${dataset_lc}_bs${batch_size}_seed${seed}.log"
        tgl_log="${LOG_DIR}/tgl_serial_${model_lc}_${dataset_lc}_bs${batch_size}_seed${seed}.log"
        master_port=$(( 33000 + combo_idx ))

        echo
        echo "=== ${model_uc} ${dataset_uc} batch=${batch_size} seed=${seed} ==="
        echo "[FROST] launching on 1 GPU, epochs=${frost_epochs}"
        if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" \
          FROST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH="${WARMUP_BATCHES}" \
          FROST_THROUGHPUT_MEASURE_BATCHES="${MEASURE_BATCHES}" \
          uv run python "${REPO_ROOT}/scripts/train_preset.py" "${model_lc}" \
            --nproc-per-node 2 \
            --master-port "${master_port}" \
            "dataset=${dataset_uc}" \
            "epoch=${frost_epochs}" \
            "macro_batch_size=${batch_size}" \
            "stable_mode=true" \
            "seed=${seed}" \
            "compile=false" \
            "tf32=false" \
            "tqdm=true" \
            > "${frost_log}" 2>&1; then
          echo "FROST run failed. See ${frost_log}" >&2
          echo "- FAILED_FROST: ${model_lc}/${dataset_uc}/bs${batch_size}/seed${seed} -> ${frost_log}" >> "${PROGRESS_MD}"
          exit 1
        fi

        read -r frost_eps frost_measured_batches < <(parse_frost_events_per_sec_per_gpu "${frost_log}")
        echo "[FROST] events/sec/gpu=${frost_eps} measured_batches=${frost_measured_batches}"
        if (( frost_measured_batches < MEASURE_BATCHES )); then
          if (( REQUIRE_FULL_MEASURE == 1 )); then
            echo "FROST measured only ${frost_measured_batches} batches (< ${MEASURE_BATCHES}) for ${model_lc}/${dataset_uc}/bs${batch_size}" >&2
            echo "- FAILED_FROST_WINDOW: ${model_lc}/${dataset_uc}/bs${batch_size}/seed${seed} measured ${frost_measured_batches}" >> "${PROGRESS_MD}"
            exit 1
          fi
          echo "Warning: FROST measured only ${frost_measured_batches} batches (< ${MEASURE_BATCHES}) for ${model_lc}/${dataset_uc}/bs${batch_size}" >&2
        fi

        echo "[TGL serial] launching on 1 GPU with pin_memory=${TGL_PIN_MEMORY} and stable env, max_train_steps=${tgl_max_train_steps}"
        tgl_cmd=(
          uv run --project "${SCRIPT_DIR}" python "${SCRIPT_DIR}/train.py"
          --data "${dataset_uc}"
          --config "${cfg_path}"
          --gpu 0
          --seed "${seed}"
          --batch_size "${batch_size}"
          --memory_update_delay_batches 0
          --appendix-train-e2e
          --measure-start-epoch 0
          --warmup-batches "${WARMUP_BATCHES}"
          --measure-batches "${MEASURE_BATCHES}"
          --max-train-steps "${tgl_max_train_steps}"
        )
        if [[ "${TGL_PIN_MEMORY}" == "1" ]]; then
          tgl_cmd+=(--pin_memory)
        fi
        if ! CUDA_VISIBLE_DEVICES="${GPU_ID}" "${tgl_cmd[@]}" > "${tgl_log}" 2>&1; then
          echo "TGL serial run failed. See ${tgl_log}" >&2
          echo "- FAILED_TGL: ${model_lc}/${dataset_uc}/bs${batch_size}/seed${seed} -> ${tgl_log}" >> "${PROGRESS_MD}"
          exit 1
        fi

        read -r tgl_eps tgl_measured_batches < <(parse_tgl_serial_events_per_sec_per_gpu "${tgl_log}")
        echo "[TGL serial] events/sec/gpu=${tgl_eps} measured_batches=${tgl_measured_batches}"
        if (( tgl_measured_batches >= 0 && tgl_measured_batches < MEASURE_BATCHES )); then
          if (( REQUIRE_FULL_MEASURE == 1 )); then
            echo "TGL serial measured only ${tgl_measured_batches} batches (< ${MEASURE_BATCHES}) for ${model_lc}/${dataset_uc}/bs${batch_size}" >&2
            echo "- FAILED_TGL_WINDOW: ${model_lc}/${dataset_uc}/bs${batch_size}/seed${seed} measured ${tgl_measured_batches}" >> "${PROGRESS_MD}"
            exit 1
          fi
          echo "Warning: TGL serial measured only ${tgl_measured_batches} batches (< ${MEASURE_BATCHES}) for ${model_lc}/${dataset_uc}/bs${batch_size}" >&2
        fi

        speedup="$(uv run python - "${frost_eps}" "${tgl_eps}" <<'PY'
import sys
f = float(sys.argv[1])
t = float(sys.argv[2])
print(f"{(f / t) if t > 0 else float('nan'):.6f}")
PY
)"

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${model_lc}" "${dataset_uc}" "${seed}" "${GPU_ID}" "${batch_size}" \
          "${frost_epochs}" "${tgl_max_train_steps}" "${frost_eps}" "${tgl_eps}" "${speedup}" \
          "${frost_measured_batches}" "${tgl_measured_batches}" \
          "${frost_log}" "${tgl_log}" \
          >> "${RESULTS_TSV}"

        echo "[SUMMARY] FROST/TGL speedup=${speedup}x"
        echo "- DONE: ${model_lc}/${dataset_uc}/bs${batch_size}/seed${seed} -> speedup ${speedup}x" >> "${PROGRESS_MD}"
      done
    done
  done
done

uv run python - "${RESULTS_TSV}" "${SUMMARY_TSV}" "${SUMMARY_TXT}" "${REPORT_MD}" <<'PY'
import csv
import pathlib
import statistics
import sys
from collections import defaultdict

results_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
summary_txt_path = pathlib.Path(sys.argv[3])
report_md_path = pathlib.Path(sys.argv[4])

with results_path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

grouped = defaultdict(list)
model_rows = defaultdict(list)
for row in rows:
    grouped[(row["model"], row["dataset"], row["batch_size"])].append(row)
    model_rows[row["model"]].append(float(row["speedup_frost_over_tgl"]))

summary_rows = []
for key in sorted(grouped):
    items = grouped[key]
    frost = statistics.mean(float(x["frost_events_per_sec_per_gpu"]) for x in items)
    tgl = statistics.mean(float(x["tgl_serial_events_per_sec_per_gpu"]) for x in items)
    speedup = statistics.mean(float(x["speedup_frost_over_tgl"]) for x in items)
    summary_rows.append(
        {
            "model": key[0],
            "dataset": key[1],
            "batch_size": key[2],
            "runs": str(len(items)),
            "frost_events_per_sec_per_gpu_mean": f"{frost:.6f}",
            "tgl_serial_events_per_sec_per_gpu_mean": f"{tgl:.6f}",
            "speedup_frost_over_tgl_mean": f"{speedup:.6f}",
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
            "speedup_frost_over_tgl_mean",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(summary_rows)

text_lines = [
    "Single-GPU throughput summary (FROST vs TGL train.py)",
    f"Source results: {results_path}",
    "",
]
for row in summary_rows:
    text_lines.append(
        f"{row['model']} {row['dataset']} batch={row['batch_size']}: "
        f"FROST {row['frost_events_per_sec_per_gpu_mean']} ev/s/gpu, "
        f"TGL {row['tgl_serial_events_per_sec_per_gpu_mean']} ev/s/gpu, "
        f"speedup {row['speedup_frost_over_tgl_mean']}x"
    )
summary_txt_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

report_lines = [
    "# FROST 1GPU vs TGL train.py 1GPU",
    "",
    f"- Source results: `{results_path}`",
    "",
    "## By Model",
    "",
    "| Model | Speedup range | Speedup mean |",
    "| --- | ---: | ---: |",
]
for model in sorted(model_rows):
    vals = model_rows[model]
    report_lines.append(
        f"| {model.upper()} | {min(vals):.3f}x - {max(vals):.3f}x | {statistics.mean(vals):.3f}x |"
    )

report_lines += [
    "",
    "## Full Table",
    "",
    "| Model | Dataset | Batch | FROST ev/s/gpu | TGL ev/s | Speedup |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
]
for row in summary_rows:
    report_lines.append(
        f"| {row['model'].upper()} | {row['dataset']} | {row['batch_size']} | "
        f"{float(row['frost_events_per_sec_per_gpu_mean']):.1f} | "
        f"{float(row['tgl_serial_events_per_sec_per_gpu_mean']):.1f} | "
        f"{float(row['speedup_frost_over_tgl_mean']):.3f}x |"
    )
report_md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
PY

echo
echo "Finished. Results:"
echo "  ${RESULTS_TSV}"
echo "  ${SUMMARY_TSV}"
echo "  ${SUMMARY_TXT}"
echo "  ${REPORT_MD}"
echo "  ${PROGRESS_MD}"
