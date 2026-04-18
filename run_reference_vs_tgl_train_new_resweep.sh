#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU_IDS="${GPU_IDS:-0,1}"
MODELS_RAW="${MODELS:-apan jodie tgn}"
DATASETS_RAW="${DATASETS:-WIKIPEDIA REDDIT MOOC LASTFM}"
BATCH_SIZES_RAW="${BATCH_SIZES:-4000 2000 1200 600}"
SEEDS_RAW="${SEEDS:-0}"
WARMUP_BATCHES="${WARMUP_BATCHES:-5}"
MEASURE_BATCHES="${MEASURE_BATCHES:-100}"
TGL_EPOCHS="${TGL_EPOCHS:-1000}"
MAX_TRAIN_STEPS_OVERRIDE="${MAX_TRAIN_STEPS:-}"
TGL_STRICT_AVOID_RC="${TGL_STRICT_AVOID_RC:-0}"
OMP_THREADS="${OMP_NUM_THREADS:-8}"
MKL_THREADS="${MKL_NUM_THREADS:-8}"

legacy_reference_epochs_var='FR''OST_EPOCHS'
stable_mode_var='FR''OST_STABLE_MODE'
throughput_warmup_var='FR''OST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH'
throughput_measure_var='FR''OST_THROUGHPUT_MEASURE_BATCHES'

REFERENCE_EPOCHS="${REFERENCE_EPOCHS:-${!legacy_reference_epochs_var:-5}}"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
N_GPU="${#GPU_LIST[@]}"
if (( N_GPU < 1 )); then
  echo "No valid GPU ids in GPU_IDS=${GPU_IDS}" >&2
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
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/experiments/com_throughput_with_tgl_train_new/resweep_2gpu_${STAMP}}"
LOG_DIR="${RUN_DIR}/logs"
CFG_DIR="${RUN_DIR}/configs"
mkdir -p "${LOG_DIR}" "${CFG_DIR}"

RESULTS_TSV="${RUN_DIR}/results.tsv"
SUMMARY_TSV="${RUN_DIR}/summary.tsv"
PROGRESS_MD="${RUN_DIR}/progress.md"

if [[ -z "${CUDA_DEVICE_MAX_CONNECTIONS+x}" ]]; then
  export CUDA_DEVICE_MAX_CONNECTIONS=1
fi
if [[ -z "${CUBLAS_WORKSPACE_CONFIG+x}" ]]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
export "${stable_mode_var}"=1

base_tgl_config_for_model() {
  local model="${1,,}"
  case "${model}" in
    apan) echo "${SCRIPT_DIR}/config/APAN.yml" ;;
    jodie) echo "${SCRIPT_DIR}/config/JODIE.yml" ;;
    tgn) echo "${SCRIPT_DIR}/config/TGN.yml" ;;
    *)
      echo "Unsupported model=${1}" >&2
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

compute_tgl_max_train_steps() {
  local dataset="$1"
  local per_gpu_batch_size="$2"
  local n_gpu="$3"
  local warmup="$4"
  local target="$5"
  uv run python - "$dataset" "$per_gpu_batch_size" "$n_gpu" "$warmup" "$target" <<'PY'
import math
import pandas as pd
import sys
from pathlib import Path

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
        print(steps)
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
pat = re.compile(
    r"TRAIN_THROUGHPUT_PER_GPU\s+rank=(\d+)\s+events_per_sec_per_gpu=([0-9eE+\-.]+).*?measured_batches=([0-9]+)"
)
rows = [(int(m.group(1)), float(m.group(2)), int(m.group(3))) for m in pat.finditer(text)]
if not rows:
    raise SystemExit(f"Could not parse reference throughput from {sys.argv[1]}")
by_rank = {}
for rank, eps, measured in rows:
    by_rank[rank] = (eps, measured)
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
print(f"{min(eps_values):.6f}\t{min(interval_values) if interval_values else -1}")
PY
}

printf "model\tdataset\tseed\tn_gpu\tmacro_batch_size\tper_gpu_batch_size\treference_events_per_sec_per_gpu\ttgl_train_new_events_per_sec_per_gpu\tspeedup_reference_over_tgl_train_new\treference_measured_batches\ttgl_measured_batches\treference_log\ttgl_log\n" > "${RESULTS_TSV}"

cat > "${PROGRESS_MD}" <<EOF
# reference vs TGL train_new.py Resweep

- Run dir: ${RUN_DIR}
- GPU_IDS: ${GPU_IDS}
- Models: ${MODELS[*]}
- Datasets: ${DATASETS[*]}
- Macro batch sizes: ${BATCH_SIZES[*]}
- Seeds: ${SEEDS[*]}
- OMP_NUM_THREADS: ${OMP_THREADS}
- MKL_NUM_THREADS: ${MKL_THREADS}
- TGL strict_avoid_rc: ${TGL_STRICT_AVOID_RC}
- Throughput aggregation: bottleneck rank (minimum per-GPU throughput across worker ranks)

## Progress

EOF

echo "Run dir: ${RUN_DIR}"
echo "GPU_IDS: ${GPU_IDS} (n_gpu=${N_GPU})"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "OMP_NUM_THREADS=${OMP_THREADS}, MKL_NUM_THREADS=${MKL_THREADS}"

for model in "${MODELS[@]}"; do
  model_lc="${model,,}"
  model_uc="${model_lc^^}"
  src_cfg="$(base_tgl_config_for_model "${model_lc}")"

  for dataset in "${DATASETS[@]}"; do
    dataset_uc="${dataset^^}"
    dataset_lc="${dataset,,}"

    for macro_batch_size in "${BATCH_SIZES[@]}"; do
      if (( macro_batch_size % N_GPU != 0 )); then
        echo "macro_batch_size=${macro_batch_size} is not divisible by n_gpu=${N_GPU}" >&2
        exit 1
      fi
      per_gpu_batch_size="$(( macro_batch_size / N_GPU ))"
      tgl_cfg="${CFG_DIR}/${model_lc}_${dataset_lc}_mb${macro_batch_size}.yml"
      write_tgl_config "${src_cfg}" "${tgl_cfg}" "${TGL_EPOCHS}" "${per_gpu_batch_size}"

      for seed in "${SEEDS[@]}"; do
        reference_log="${LOG_DIR}/reference_${model_lc}_${dataset_lc}_mb${macro_batch_size}_seed${seed}.log"
        tgl_log="${LOG_DIR}/tgl_train_new_${model_lc}_${dataset_lc}_mb${macro_batch_size}_seed${seed}.log"

        echo
        echo "=== ${model_uc} ${dataset_uc} mb=${macro_batch_size} seed=${seed} ==="

        "${throughput_warmup_var}=${WARMUP_BATCHES}" \
        "${throughput_measure_var}=${MEASURE_BATCHES}" \
        OMP_NUM_THREADS="${OMP_THREADS}" \
        MKL_NUM_THREADS="${MKL_THREADS}" \
        CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
        uv run python scripts/train_preset.py "${model_lc}" \
          --nproc-per-node "$((N_GPU + 1))" \
          --master-port $((29660 + seed)) \
          dataset="${dataset_uc}" \
          epoch="${REFERENCE_EPOCHS}" \
          macro_batch_size="${macro_batch_size}" \
          stable_mode=true \
          compile=false \
          tf32=false \
          seed="${seed}" \
          tqdm=true \
          > "${reference_log}" 2>&1

        if [[ -n "${MAX_TRAIN_STEPS_OVERRIDE}" ]]; then
          tgl_max_train_steps="${MAX_TRAIN_STEPS_OVERRIDE}"
        else
          tgl_max_train_steps="$(compute_tgl_max_train_steps "${dataset_uc}" "${per_gpu_batch_size}" "${N_GPU}" "${WARMUP_BATCHES}" "${MEASURE_BATCHES}")"
        fi

        strict_rc_args=()
        if [[ "${TGL_STRICT_AVOID_RC}" == "1" ]]; then
          strict_rc_args+=(--strict_avoid_rc)
        fi

        OMP_NUM_THREADS="${OMP_THREADS}" \
        MKL_NUM_THREADS="${MKL_THREADS}" \
        CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
        uv run --project "${SCRIPT_DIR}" python "${SCRIPT_DIR}/train_new.py" \
          --dataset "${dataset_lc}" \
          --config "${tgl_cfg}" \
          --num_gpus "${N_GPU}" \
          --seed "${seed}" \
          --rnd_edim 0 \
          --rnd_ndim 0 \
          --appendix-train-e2e \
          --measure-start-epoch 0 \
          --warmup-batches "${WARMUP_BATCHES}" \
          --measure-batches "${MEASURE_BATCHES}" \
          --max-train-steps "${tgl_max_train_steps}" \
          --tqdm \
          "${strict_rc_args[@]}" \
          > "${tgl_log}" 2>&1

        read -r reference_eps reference_measured_batches < <(parse_reference_events_per_sec_per_gpu "${reference_log}")
        read -r tgl_eps tgl_measured_batches < <(parse_tgl_events_per_sec_per_gpu "${tgl_log}")

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

        printf -- "- [%s / %s / mb=%s / seed=%s] reference=%s, TGL=%s, speedup=%sx\n" \
          "${model_uc}" "${dataset_uc}" "${macro_batch_size}" "${seed}" \
          "${reference_eps}" "${tgl_eps}" "${speedup}" \
          >> "${PROGRESS_MD}"
      done
    done
  done
done

uv run python - "${RESULTS_TSV}" "${SUMMARY_TSV}" <<'PY'
import csv
import pathlib
import statistics
import sys
from collections import defaultdict

results_tsv = pathlib.Path(sys.argv[1])
summary_tsv = pathlib.Path(sys.argv[2])
rows = list(csv.DictReader(results_tsv.read_text().splitlines(), delimiter="\t"))
group = defaultdict(list)
for r in rows:
    key = (r["model"], r["dataset"], r["macro_batch_size"])
    group[key].append(r)

summary_rows = []
for (model, dataset, macro_batch_size), items in sorted(group.items()):
    reference = statistics.mean(float(x["reference_events_per_sec_per_gpu"]) for x in items)
    tgl = statistics.mean(float(x["tgl_train_new_events_per_sec_per_gpu"]) for x in items)
    speedup = reference / tgl if tgl > 0 else float("nan")
    summary_rows.append(
        {
            "model": model,
            "dataset": dataset,
            "macro_batch_size": macro_batch_size,
            "runs": str(len(items)),
            "reference_events_per_sec_per_gpu_mean": f"{reference:.6f}",
            "tgl_train_new_events_per_sec_per_gpu_mean": f"{tgl:.6f}",
            "speedup_reference_over_tgl_train_new_mean": f"{speedup:.6f}",
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
            "tgl_train_new_events_per_sec_per_gpu_mean",
            "speedup_reference_over_tgl_train_new_mean",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(summary_rows)
PY

echo "Results TSV: ${RESULTS_TSV}"
echo "Summary TSV: ${SUMMARY_TSV}"
