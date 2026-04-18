#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

GPU_IDS="${GPU_IDS:-0,1}"
MODELS_RAW="${MODELS:-apan jodie tgn}"
DATASETS_RAW="${DATASETS:-LASTFM MOOC REDDIT WIKIPEDIA}"
SEEDS_RAW="${SEEDS:-0}"
BATCH_SIZES_RAW="${BATCH_SIZES:-600 1200 2000 4000}"
WARMUP_BATCHES="${WARMUP_BATCHES:-5}"
MEASURE_BATCHES="${MEASURE_BATCHES:-100}"
REQUIRE_FULL_MEASURE="${REQUIRE_FULL_MEASURE:-1}"
TGL_EPOCHS="${TGL_EPOCHS:-1000}"
MAX_TRAIN_STEPS_OVERRIDE="${MAX_TRAIN_STEPS:-}"
TGL_STRICT_AVOID_RC="${TGL_STRICT_AVOID_RC:-0}"

if (( WARMUP_BATCHES < 0 )); then
  echo "WARMUP_BATCHES must be >= 0, got ${WARMUP_BATCHES}" >&2
  exit 1
fi
if (( MEASURE_BATCHES < 1 )); then
  echo "MEASURE_BATCHES must be >= 1, got ${MEASURE_BATCHES}" >&2
  exit 1
fi
if [[ "${TGL_STRICT_AVOID_RC}" != "0" && "${TGL_STRICT_AVOID_RC}" != "1" ]]; then
  echo "TGL_STRICT_AVOID_RC must be 0 or 1, got ${TGL_STRICT_AVOID_RC}" >&2
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
BASE_DIR="${REPO_ROOT}/experiments/com_throughput_with_tgl_train_new"
if [[ "${TGL_STRICT_AVOID_RC}" == "1" ]]; then
  DEFAULT_RUN_BASENAME="throughput_2gpu_train_new_pinmem_strictrc_${STAMP}"
else
  DEFAULT_RUN_BASENAME="throughput_2gpu_train_new_pinmem_${STAMP}"
fi
RUN_DIR="${RUN_DIR:-${BASE_DIR}/${DEFAULT_RUN_BASENAME}}"
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
stable_mode_var='FR''OST_STABLE_MODE'
export "${stable_mode_var}"=1

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

reference_summary_for_model() {
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

lookup_reference_summary_row() {
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
            metric = row.get("reference_events_per_sec_per_gpu_mean")
            if metric is None:
                legacy_prefix = "fr" "ost"
                metric = row[f"{legacy_prefix}_events_per_sec_per_gpu_mean"]
            print("\t".join([metric, str(row.get("runs", "1"))]))
            raise SystemExit(0)
raise SystemExit(
    f"Could not find reference summary row for model={model} dataset={dataset} macro_batch_size={macro_batch_size} in {summary_path}"
)
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

printf "model\tdataset\tseed\tn_gpu\tmacro_batch_size\tper_gpu_batch_size\treference_events_per_sec_per_gpu\ttgl_train_new_events_per_sec_per_gpu\tspeedup_reference_over_tgl_train_new\treference_reference_runs\ttgl_measured_batches\treference_summary_path\ttgl_log\n" > "${RESULTS_TSV}"

cat > "${PROGRESS_MD}" <<EOF
# TGL train_new.py vs Existing reference 2GPU Throughput

- Run dir: ${RUN_DIR}
- GPU_IDS: ${GPU_IDS}
- Models: ${MODELS[*]}
- Datasets: ${DATASETS[*]}
- Macro batch sizes: ${BATCH_SIZES[*]}
- Seeds: ${SEEDS[*]}
- TGL script: \`third_party/tgl/train_new.py\`
- train_new pinned buffers: always enabled in the current implementation (equivalent to pin_memory=true)
- train_new synchronization: next-step sampling starts only after the previous batch finishes \`update_mailbox/update_memory\`
- train_new strict_avoid_rc: ${TGL_STRICT_AVOID_RC}
- Measurement: skip ${WARMUP_BATCHES} warm-up train intervals, measure next ${MEASURE_BATCHES} train intervals
- Max train steps per run: auto-computed per dataset/batch combo${MAX_TRAIN_STEPS_OVERRIDE:+ (override=${MAX_TRAIN_STEPS_OVERRIDE})}
- Stable env: \`CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}\`, \`CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}\`
- reference reference: existing 2-GPU fair summaries under \`experiments/com_throughput_with_tgl/\`

## Progress

EOF

cat > "${REPORT_MD}" <<EOF
# reference vs TGL train_new.py (2 GPU)

- TGL runner: \`third_party/tgl/train_new.py\`
- GPU count: ${N_GPU}
- train_new pinned buffers: enabled by default in code, so this run matches \`pin_mem=true\`
- train_new synchronization: barrier before host-side next-step sampling
- train_new strict_avoid_rc: ${TGL_STRICT_AVOID_RC}
- Measurement: skip ${WARMUP_BATCHES} warm-up intervals, measure ${MEASURE_BATCHES} intervals
- Max train steps: auto-computed per dataset/batch combo${MAX_TRAIN_STEPS_OVERRIDE:+ (override=${MAX_TRAIN_STEPS_OVERRIDE})}
- Stable env: \`CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}\`, \`CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}\`
- reference values are read from existing 2-GPU fair summaries, not rerun here

EOF

echo "=== TGL train_new.py Throughput Benchmark ==="
echo "Run dir: ${RUN_DIR}"
echo "GPU_IDS: ${GPU_IDS} (n_gpu=${N_GPU})"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Macro batch sizes: ${BATCH_SIZES[*]}"
echo "Stable env: CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}, CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}, stable_mode=${!stable_mode_var}"
echo "train_new pinned buffers: enabled by default"
echo "train_new strict_avoid_rc: ${TGL_STRICT_AVOID_RC}"
echo "Warmup batches: ${WARMUP_BATCHES}"
echo "Measure batches: ${MEASURE_BATCHES}"
echo "Max train steps: auto-computed per dataset/batch combo${MAX_TRAIN_STEPS_OVERRIDE:+ (override=${MAX_TRAIN_STEPS_OVERRIDE})}"
echo "Require full measurement window: ${REQUIRE_FULL_MEASURE}"

combo_idx=0
for model in "${MODELS[@]}"; do
  model_lc="${model,,}"
  model_uc="${model_lc^^}"
  src_cfg="$(base_tgl_config_for_model "${model_lc}")"
  reference_summary="$(reference_summary_for_model "${model_lc}")"

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
      write_tgl_config "${src_cfg}" "${tgl_cfg}" "${TGL_EPOCHS}" "${per_gpu_batch_size}"
      read -r reference_eps reference_reference_runs < <(lookup_reference_summary_row "${reference_summary}" "${model_lc}" "${dataset_uc}" "${macro_batch_size}")

      for seed in "${SEEDS[@]}"; do
        combo_idx=$(( combo_idx + 1 ))
        tgl_log="${LOG_DIR}/tgl_train_new_${model_lc}_${dataset_lc}_mb${macro_batch_size}_seed${seed}.log"

        echo
        echo "=== ${model_uc} ${dataset_uc} mb=${macro_batch_size} seed=${seed} ==="
        echo "macro_batch_size=${macro_batch_size}, per_gpu_batch_size=${per_gpu_batch_size}"
        echo "reference reference events/sec/gpu=${reference_eps}"
        if [[ -n "${MAX_TRAIN_STEPS_OVERRIDE}" ]]; then
          tgl_max_train_steps="${MAX_TRAIN_STEPS_OVERRIDE}"
        else
          tgl_max_train_steps="$(compute_tgl_max_train_steps "${dataset_uc}" "${per_gpu_batch_size}" "${N_GPU}" "${WARMUP_BATCHES}" "${MEASURE_BATCHES}")"
        fi
        strict_rc_args=()
        if [[ "${TGL_STRICT_AVOID_RC}" == "1" ]]; then
          strict_rc_args+=(--strict_avoid_rc)
        fi

        echo "[TGL train_new] launching on ${N_GPU} GPUs with pinned buffers enabled and stable env, max_train_steps=${tgl_max_train_steps}"

        if ! CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
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
            > "${tgl_log}" 2>&1; then
          echo "TGL train_new run failed. See ${tgl_log}" >&2
          exit 1
        fi

        read -r tgl_eps tgl_measured_batches < <(parse_tgl_events_per_sec_per_gpu "${tgl_log}")
        echo "[TGL train_new] events/sec/gpu=${tgl_eps} measured_batches=${tgl_measured_batches}"
        if (( tgl_measured_batches >= 0 && tgl_measured_batches < MEASURE_BATCHES )); then
          if (( REQUIRE_FULL_MEASURE == 1 )); then
            echo "TGL train_new measured only ${tgl_measured_batches} batches (< ${MEASURE_BATCHES}) for ${model_lc}/${dataset_uc}/mb${macro_batch_size}" >&2
            exit 1
          fi
          echo "Warning: TGL train_new measured only ${tgl_measured_batches} batches (< ${MEASURE_BATCHES}) for ${model_lc}/${dataset_uc}/mb${macro_batch_size}" >&2
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
          "${reference_reference_runs}" "${tgl_measured_batches}" \
          "${reference_summary}" "${tgl_log}" \
          >> "${RESULTS_TSV}"

        {
          printf -- "- [%s / %s / mb=%s / seed=%s] reference=%s ev/s/gpu, TGL train_new=%s ev/s/gpu, speedup=%sx\n" \
            "${model_uc}" "${dataset_uc}" "${macro_batch_size}" "${seed}" "${reference_eps}" "${tgl_eps}" "${speedup}"
        } >> "${PROGRESS_MD}"

        echo "[SUMMARY] speedup_reference_over_tgl_train_new=${speedup}x"
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

results_tsv = pathlib.Path(sys.argv[1])
summary_tsv = pathlib.Path(sys.argv[2])
summary_txt = pathlib.Path(sys.argv[3])
report_md = pathlib.Path(sys.argv[4])

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

lines = ["model\tdataset\tmacro_batch_size\truns\treference_eps_mean\ttgl_train_new_eps_mean\tspeedup_mean"]
for r in summary_rows:
    lines.append(
        f"{r['model']}\t{r['dataset']}\t{r['macro_batch_size']}\t{r['runs']}\t"
        f"{r['reference_events_per_sec_per_gpu_mean']}\t"
        f"{r['tgl_train_new_events_per_sec_per_gpu_mean']}\t"
        f"{r['speedup_reference_over_tgl_train_new_mean']}"
    )
summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

report_lines = report_md.read_text(encoding="utf-8").rstrip().splitlines()
report_lines.extend(
    [
        "",
        "## Summary",
        "",
        "| model | dataset | macro_batch_size | reference ev/s/gpu | TGL train_new ev/s/gpu | speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
)
for r in summary_rows:
    report_lines.append(
        f"| {r['model']} | {r['dataset']} | {r['macro_batch_size']} | "
        f"{r['reference_events_per_sec_per_gpu_mean']} | "
        f"{r['tgl_train_new_events_per_sec_per_gpu_mean']} | "
        f"{r['speedup_reference_over_tgl_train_new_mean']}x |"
    )
report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
PY

echo
echo "Results TSV: ${RESULTS_TSV}"
echo "Summary TSV: ${SUMMARY_TSV}"
echo "Summary TXT: ${SUMMARY_TXT}"
echo "Report MD: ${REPORT_MD}"
cat "${SUMMARY_TXT}"
