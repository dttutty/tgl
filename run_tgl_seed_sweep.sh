#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

GPU_IDS="${GPU_IDS:-0,1}"
MODELS_RAW="${MODELS:-apan tgn}"
DATASETS_RAW="${DATASETS:-LASTFM MOOC REDDIT WIKIPEDIA}"
SEEDS_RAW="${SEEDS:-0}"
BATCH_SIZES_RAW="${BATCH_SIZES:-600 1200 2000 4000}"

WARMUP_BATCHES="${WARMUP_BATCHES:-5}"
MEASURE_BATCHES="${MEASURE_BATCHES:-100}"
REQUIRE_FULL_MEASURE="${REQUIRE_FULL_MEASURE:-1}"
MAX_TRAIN_STEPS_OVERRIDE="${MAX_TRAIN_STEPS:-}"
OMP_THREADS="${OMP_NUM_THREADS:-8}"
MKL_THREADS="${MKL_NUM_THREADS:-8}"
legacy_reference_epochs_var='FR''OST_EPOCHS'
stable_mode_var='FR''OST_STABLE_MODE'
throughput_warmup_var='FR''OST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH'
throughput_measure_var='FR''OST_THROUGHPUT_MEASURE_BATCHES'
REFERENCE_EPOCHS="${REFERENCE_EPOCHS:-${!legacy_reference_epochs_var:-1}}"
if [[ -n "${TGL_EPOCHS:-}" ]]; then
  TGL_EPOCHS="${TGL_EPOCHS}"
else
  TGL_EPOCHS=1000
fi

if (( WARMUP_BATCHES < 0 )); then
  echo "WARMUP_BATCHES must be >= 0, got ${WARMUP_BATCHES}" >&2
  exit 1
fi
if (( MEASURE_BATCHES < 1 )); then
  echo "MEASURE_BATCHES must be >= 1, got ${MEASURE_BATCHES}" >&2
  exit 1
fi
if [[ -n "${MAX_TRAIN_STEPS_OVERRIDE}" ]] && (( MAX_TRAIN_STEPS_OVERRIDE < 0 )); then
  echo "MAX_TRAIN_STEPS must be >= 0, got ${MAX_TRAIN_STEPS_OVERRIDE}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
N_GPU="${#GPU_LIST[@]}"
if (( N_GPU < 1 )); then
  echo "No valid GPU ids in GPU_IDS=${GPU_IDS}" >&2
  exit 1
fi
if (( N_GPU != 2 )); then
  echo "Warning: expected 2 GPUs for this benchmark, but got ${N_GPU} from GPU_IDS=${GPU_IDS}" >&2
fi

MODELS_NORM="${MODELS_RAW//,/ }"
DATASETS_NORM="${DATASETS_RAW//,/ }"
SEEDS_NORM="${SEEDS_RAW//,/ }"
BATCH_SIZES_NORM="${BATCH_SIZES_RAW//,/ }"
IFS=' ' read -r -a MODELS <<< "${MODELS_NORM}"
IFS=' ' read -r -a DATASETS <<< "${DATASETS_NORM}"
IFS=' ' read -r -a SEEDS <<< "${SEEDS_NORM}"
IFS=' ' read -r -a BATCH_SIZES <<< "${BATCH_SIZES_NORM}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/experiments/com_throughput_with_tgl/throughput_2gpu_${STAMP}}"
LOG_DIR="${RUN_DIR}/logs"
CFG_DIR="${RUN_DIR}/configs"
mkdir -p "${LOG_DIR}" "${CFG_DIR}"

RESULTS_TSV="${RUN_DIR}/results.tsv"
SUMMARY_TSV="${RUN_DIR}/summary.tsv"
SUMMARY_TXT="${RUN_DIR}/summary.txt"

if [[ -z "${CUDA_DEVICE_MAX_CONNECTIONS+x}" ]]; then
  export CUDA_DEVICE_MAX_CONNECTIONS=1
fi
if [[ -z "${CUBLAS_WORKSPACE_CONFIG+x}" ]]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
export "${stable_mode_var}"=1

tgl_config_for_model() {
  local model="${1,,}"
  case "${model}" in
    tgn) echo "${SCRIPT_DIR}/config/TGN.yml" ;;
    jodie) echo "${SCRIPT_DIR}/config/JODIE.yml" ;;
    apan) echo "${SCRIPT_DIR}/config/APAN.yml" ;;
    *)
      echo "Unsupported model=${1}. Supported: tgn, jodie, apan" >&2
      return 1
      ;;
  esac
}

compute_tgl_max_train_steps() {
  local dataset="$1"
  local per_gpu_batch_size="$2"
  local n_gpu="$3"
  local warmup="$4"
  local target="$5"
  uv run python - "$dataset" "$per_gpu_batch_size" "$n_gpu" "$warmup" "$target" <<'PY'
import math
from pathlib import Path
import sys

import pandas as pd

dataset = sys.argv[1].upper()
per_gpu_batch_size = int(sys.argv[2])
n_gpu = int(sys.argv[3])
warmup = int(sys.argv[4])
target = int(sys.argv[5])

df = pd.read_csv(Path("DATA") / dataset / "edges.csv")
train_edge_end = int(df[df["default_split"].gt(0)].index[0])
steps_per_epoch = math.ceil(train_edge_end / (per_gpu_batch_size * n_gpu))
required_intervals = warmup + target

steps = 1
while True:
    full_epochs, partial_steps = divmod(steps, steps_per_epoch)
    intervals = full_epochs * max(steps_per_epoch - 1, 0)
    if partial_steps > 0:
        intervals += max(partial_steps - 1, 0)
    if intervals >= required_intervals:
        # train_dist.py can end up one train-batch short in its final
        # appendix accounting, so keep a tiny safety margin here.
        print(steps + 2)
        raise SystemExit(0)
    steps += 1
PY
}

parse_reference_events_per_sec_per_gpu() {
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
    raise SystemExit(f"Could not parse reference throughput from {sys.argv[1]}")
by_rank = {}
for rank_s, eps_s, measured_s in matches:
    by_rank[int(rank_s)] = (float(eps_s), int(measured_s))
eps_values = [value[0] for value in by_rank.values()]
measured_values = [value[1] for value in by_rank.values()]
print(f"{min(eps_values):.6f}\t{min(measured_values)}")
PY
}

parse_tgl_events_per_sec_per_gpu() {
  local log_path="$1"
  uv run python - "$log_path" <<'PY'
import json
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
rows = []
decoder = json.JSONDecoder()
for match in re.finditer(r"APPENDIX_TRAIN_E2E_SUMMARY\s+", text):
    start = match.end()
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        continue
    v = obj.get("events_per_sec_per_gpu")
    if not isinstance(v, (int, float)):
        continue
    rank = obj.get("rank")
    interval_count = obj.get("interval_count")
    rank_key = int(rank) if isinstance(rank, int) else len(rows)
    rows.append((rank_key, float(v), int(interval_count) if isinstance(interval_count, int) else -1))
if not rows:
    raise SystemExit(f"Could not parse TGL throughput from {sys.argv[1]}")
by_rank = {}
for rank_key, eps, interval_count in rows:
    by_rank[rank_key] = (eps, interval_count)
eps_values = [value[0] for value in by_rank.values()]
interval_values = [value[1] for value in by_rank.values() if value[1] >= 0]
min_interval = min(interval_values) if interval_values else -1
print(f"{min(eps_values):.6f}\t{min_interval}")
PY
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

printf "model\tdataset\tseed\tn_gpu\tmacro_batch_size\tper_gpu_batch_size\treference_events_per_sec_per_gpu\ttgl_events_per_sec_per_gpu\tspeedup_reference_over_tgl\treference_measured_batches\ttgl_measured_batches\treference_log\ttgl_log\n" > "${RESULTS_TSV}"

echo "=== Throughput Benchmark (reference vs TGL) ==="
echo "Run dir: ${RUN_DIR}"
echo "GPU_IDS: ${GPU_IDS} (n_gpu=${N_GPU})"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Macro batch sizes: ${BATCH_SIZES[*]}"
echo "Stable mode env: CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}, CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}, stable_mode=${!stable_mode_var}"
echo "reference: epoch=${REFERENCE_EPOCHS} (uses built-in TRAIN_THROUGHPUT_PER_GPU line)"
echo "TGL: epoch=${TGL_EPOCHS}, warmup_batches=${WARMUP_BATCHES}, measure_batches=${MEASURE_BATCHES}, max_train_steps=dynamic"
echo "OMP_NUM_THREADS=${OMP_THREADS}, MKL_NUM_THREADS=${MKL_THREADS}"
echo "Aggregation: bottleneck rank (min events/sec/gpu)"
echo "Require full measurement window: ${REQUIRE_FULL_MEASURE}"

combo_idx=0
for model in "${MODELS[@]}"; do
  model_lc="${model,,}"
  tgl_src_cfg="$(tgl_config_for_model "${model_lc}")"

  for dataset in "${DATASETS[@]}"; do
    dataset_uc="${dataset^^}"
    dataset_lc="${dataset,,}"
    for macro_batch_size in "${BATCH_SIZES[@]}"; do
      if (( macro_batch_size < 1 )); then
        echo "Invalid macro_batch_size=${macro_batch_size}" >&2
        exit 1
      fi
      if (( macro_batch_size % N_GPU != 0 )); then
        echo "macro_batch_size=${macro_batch_size} is not divisible by n_gpu=${N_GPU}" >&2
        exit 1
      fi
      per_gpu_batch_size="$(( macro_batch_size / N_GPU ))"
      tgl_cfg="${CFG_DIR}/${model_lc}_${dataset_lc}_mb${macro_batch_size}.yml"
      write_tgl_config "${tgl_src_cfg}" "${tgl_cfg}" "${TGL_EPOCHS}" "${per_gpu_batch_size}"

      for seed in "${SEEDS[@]}"; do
        combo_idx=$((combo_idx + 1))
        reference_log="${LOG_DIR}/reference_${model_lc}_${dataset_lc}_mb${macro_batch_size}_seed${seed}.log"
        tgl_log="${LOG_DIR}/tgl_${model_lc}_${dataset_lc}_mb${macro_batch_size}_seed${seed}.log"
        master_port=$(( 32000 + combo_idx ))

        echo
        echo "=== ${model_lc^^} ${dataset_uc} mb=${macro_batch_size} seed=${seed} ==="
        echo "macro_batch_size=${macro_batch_size}, per_gpu_batch_size=${per_gpu_batch_size}"

        echo "[reference] launching..."
        if ! OMP_NUM_THREADS="${OMP_THREADS}" \
          MKL_NUM_THREADS="${MKL_THREADS}" \
          CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
          "${throughput_warmup_var}=${WARMUP_BATCHES}" \
          "${throughput_measure_var}=${MEASURE_BATCHES}" \
          uv run python "${REPO_ROOT}/scripts/train_preset.py" "${model_lc}" \
            --nproc-per-node "$(( N_GPU + 1 ))" \
            --master-port "${master_port}" \
            "dataset=${dataset_uc}" \
            "epoch=${REFERENCE_EPOCHS}" \
            "macro_batch_size=${macro_batch_size}" \
            "stable_mode=true" \
            "seed=${seed}" \
            "compile=false" \
            "tf32=false" \
            "tqdm=true" \
            > "${reference_log}" 2>&1; then
          echo "reference run failed. See ${reference_log}" >&2
          exit 1
        fi
        read -r reference_eps reference_measured_batches < <(parse_reference_events_per_sec_per_gpu "${reference_log}")
        echo "[reference] events/sec/gpu=${reference_eps} measured_batches=${reference_measured_batches}"
        if (( reference_measured_batches < MEASURE_BATCHES )); then
          if (( REQUIRE_FULL_MEASURE == 1 )); then
            echo "reference measured only ${reference_measured_batches} batches (< ${MEASURE_BATCHES}) for ${model_lc}/${dataset_uc}/mb${macro_batch_size}" >&2
            exit 1
          fi
          echo "Warning: reference measured only ${reference_measured_batches} batches (< ${MEASURE_BATCHES}) for ${model_lc}/${dataset_uc}/mb${macro_batch_size}" >&2
        fi

        echo "[TGL] launching..."
        if [[ -n "${MAX_TRAIN_STEPS_OVERRIDE}" ]]; then
          tgl_max_train_steps="${MAX_TRAIN_STEPS_OVERRIDE}"
        else
          tgl_max_train_steps="$(compute_tgl_max_train_steps "${dataset_uc}" "${per_gpu_batch_size}" "${N_GPU}" "${WARMUP_BATCHES}" "${MEASURE_BATCHES}")"
        fi
        tgl_cmd=(
          uv run --project "${SCRIPT_DIR}" python "${SCRIPT_DIR}/train_dist.py"
          --dataset "${dataset_lc}"
          --config "${tgl_cfg}"
          --num_gpus "${N_GPU}"
          --seed "${seed}"
          --rnd_edim 0
          --rnd_ndim 0
          --appendix-train-e2e
          --measure-start-epoch 0
          --warmup-batches "${WARMUP_BATCHES}"
          --measure-batches "${MEASURE_BATCHES}"
          --tqdm
        )
        if (( tgl_max_train_steps > 0 )); then
          tgl_cmd+=(--max-train-steps "${tgl_max_train_steps}")
        fi
        echo "[TGL] max_train_steps=${tgl_max_train_steps}"
        if ! OMP_NUM_THREADS="${OMP_THREADS}" \
          MKL_NUM_THREADS="${MKL_THREADS}" \
          CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${tgl_cmd[@]}" > "${tgl_log}" 2>&1; then
          echo "TGL run failed. See ${tgl_log}" >&2
          exit 1
        fi
        read -r tgl_eps tgl_measured_batches < <(parse_tgl_events_per_sec_per_gpu "${tgl_log}")
        echo "[TGL] events/sec/gpu=${tgl_eps} measured_batches=${tgl_measured_batches}"
        if (( tgl_measured_batches >= 0 && tgl_measured_batches < MEASURE_BATCHES )); then
          if (( REQUIRE_FULL_MEASURE == 1 )); then
            echo "TGL measured only ${tgl_measured_batches} batches (< ${MEASURE_BATCHES}) for ${model_lc}/${dataset_uc}/mb${macro_batch_size}" >&2
            exit 1
          fi
          echo "Warning: TGL measured only ${tgl_measured_batches} batches (< ${MEASURE_BATCHES}) for ${model_lc}/${dataset_uc}/mb${macro_batch_size}" >&2
        fi

        speedup="$(uv run python - "${reference_eps}" "${tgl_eps}" <<'PY'
import sys
f = float(sys.argv[1])
t = float(sys.argv[2])
print(f"{(f / t) if t > 0 else float('nan'):.6f}")
PY
)"

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "${model_lc}" "${dataset_uc}" "${seed}" "${N_GPU}" \
          "${macro_batch_size}" "${per_gpu_batch_size}" \
          "${reference_eps}" "${tgl_eps}" "${speedup}" \
          "${reference_measured_batches}" "${tgl_measured_batches}" \
          "${reference_log}" "${tgl_log}" \
          >> "${RESULTS_TSV}"

        echo "[SUMMARY] speedup_reference_over_tgl=${speedup}x"
      done
    done
  done
done

uv run python - "${RESULTS_TSV}" "${SUMMARY_TSV}" "${SUMMARY_TXT}" <<'PY'
import csv
import pathlib
import statistics
import sys
from collections import defaultdict

results_tsv = pathlib.Path(sys.argv[1])
summary_tsv = pathlib.Path(sys.argv[2])
summary_txt = pathlib.Path(sys.argv[3])

rows = list(csv.DictReader(results_tsv.read_text().splitlines(), delimiter="\t"))
group = defaultdict(list)
for r in rows:
    key = (r["model"], r["dataset"], r["macro_batch_size"])
    group[key].append(r)

summary_rows = []
for (model, dataset, macro_batch_size), items in sorted(group.items()):
    reference = statistics.mean(float(x["reference_events_per_sec_per_gpu"]) for x in items)
    tgl = statistics.mean(float(x["tgl_events_per_sec_per_gpu"]) for x in items)
    speedup = reference / tgl if tgl > 0 else float("nan")
    summary_rows.append(
        {
            "model": model,
            "dataset": dataset,
            "macro_batch_size": macro_batch_size,
            "runs": str(len(items)),
            "reference_events_per_sec_per_gpu_mean": f"{reference:.6f}",
            "tgl_events_per_sec_per_gpu_mean": f"{tgl:.6f}",
            "speedup_reference_over_tgl_mean": f"{speedup:.6f}",
        }
    )

with summary_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "dataset",
            "macro_batch_size",
            "runs",
            "reference_events_per_sec_per_gpu_mean",
            "tgl_events_per_sec_per_gpu_mean",
            "speedup_reference_over_tgl_mean",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(summary_rows)

lines = ["model\tdataset\tmacro_batch_size\truns\treference_eps_mean\ttgl_eps_mean\tspeedup_mean"]
for r in summary_rows:
    lines.append(
        f"{r['model']}\t{r['dataset']}\t{r['macro_batch_size']}\t{r['runs']}\t"
        f"{r['reference_events_per_sec_per_gpu_mean']}\t"
        f"{r['tgl_events_per_sec_per_gpu_mean']}\t"
        f"{r['speedup_reference_over_tgl_mean']}"
    )
summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo
echo "Results TSV: ${RESULTS_TSV}"
echo "Summary TSV: ${SUMMARY_TSV}"
echo "Summary TXT: ${SUMMARY_TXT}"
cat "${SUMMARY_TXT}"
