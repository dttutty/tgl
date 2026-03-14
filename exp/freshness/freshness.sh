#!/bin/bash
# Freshness sweep for delayed memory updates.
# Runs train_non_timing.py with pin_memory enabled and sweeps
# memory_update_delay_batches across datasets and model configs.
#
# Usage:
#   bash exp/freshness/freshness.sh [GPU_IDS]
#
# GPU_IDS: comma-separated list of GPU indices, e.g. "0,1,2,3,4,5,6,7" (default: all 8)
#
# Optional env vars for concurrent scheduling:
#   MAX_CONCURRENT_JOBS (default: 10)
#   EST_MEM_PER_JOB_MB (default: 5000)
#   MIN_FREE_MEM_MB (default: 1500)
#   SCHED_POLL_SECS (default: 10)

set -uo pipefail

GPUS_ARG="${1:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPUS_ARG"
NUM_GPUS="${#GPUS[@]}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$SCRIPT_DIR/logs"
TMP_CONFIG_DIR="$LOG_DIR/tmp_configs"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    # Legacy fallback for environments that rely on the original hardcoded path.
    PYTHON_BIN="/home/sqp17/miniconda3/envs/simple_py310/bin/python"
fi

mkdir -p "$LOG_DIR"
mkdir -p "$TMP_CONFIG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
    echo "Tip: set an explicit interpreter, e.g. PYTHON_BIN=\"$(command -v python3 2>/dev/null || echo /path/to/python)\" bash exp/freshness/freshness.sh" >&2
    exit 1
fi

MODELS=("TGN" "APAN" "JODIE")
CONFIGS=("config/TGN.yml" "config/APAN.yml" "config/JODIE.yml")

DATASETS=("LASTFM" "MOOC" "REDDIT" "WIKI")
EXTRA_ARGS=(
    "--rand_edge_features 128"
    "--rand_edge_features 128"
    ""
    ""
)

DELAYS=(0 1 2 3 4)
DIM_OUTS=(128 256 512)
REPEATS=20
TARGET_EPOCH=100
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-16}"
EST_MEM_PER_JOB_MB="${EST_MEM_PER_JOB_MB:-5000}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-1500}"
SCHED_POLL_SECS="${SCHED_POLL_SECS:-30}"

active_pids=()
active_descs=()
active_logs=()
active_gpus=()
started_jobs=0
finished_jobs=0

declare -A gpu_active_count
for _g in "${GPUS[@]}"; do
    gpu_active_count[$_g]=0
done
chosen_gpu="${GPUS[0]}"

get_active_count() {
    echo "${#active_pids[@]}"
}

gpu_mem_free_mb() {
    local gpu_id="${1:-0}"
    local line
    local total
    local used
    if ! line=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null); then
        echo "-1"
        return
    fi
    IFS=',' read -r total used <<< "$line"
    total="${total//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    if [[ -z "$total" || -z "$used" ]]; then
        echo "-1"
        return
    fi
    echo $((total - used))
}

reap_finished_jobs() {
    local new_pids=()
    local new_descs=()
    local new_logs=()
    local new_gpus=()
    local i
    local pid
    local desc
    local logf
    local gpuid

    for i in "${!active_pids[@]}"; do
        pid="${active_pids[$i]}"
        desc="${active_descs[$i]}"
        logf="${active_logs[$i]}"
        gpuid="${active_gpus[$i]}"
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
            new_descs+=("$desc")
            new_logs+=("$logf")
            new_gpus+=("$gpuid")
        else
            if wait "$pid"; then
                echo "[DONE] gpu=${gpuid} ${desc}"
            else
                echo "[FAILED] gpu=${gpuid} ${desc} (see ${logf})" >&2
            fi
            gpu_active_count[$gpuid]=$(( ${gpu_active_count[$gpuid]:-0} - 1 ))
            finished_jobs=$((finished_jobs + 1))
        fi
    done

    active_pids=("${new_pids[@]}")
    active_descs=("${new_descs[@]}")
    active_logs=("${new_logs[@]}")
    active_gpus=("${new_gpus[@]}")
}

wait_for_slot() {
    local need_mem=$(( EST_MEM_PER_JOB_MB + MIN_FREE_MEM_MB ))
    local max_per_gpu=$(( (MAX_CONCURRENT_JOBS + NUM_GPUS - 1) / NUM_GPUS ))

    while true; do
        reap_finished_jobs
        local total_active
        total_active=$(get_active_count)
        if [[ "$total_active" -lt "$MAX_CONCURRENT_JOBS" ]]; then
            local _g _cnt _free
            for _g in "${GPUS[@]}"; do
                _cnt="${gpu_active_count[$_g]:-0}"
                if [[ "$_cnt" -lt "$max_per_gpu" ]]; then
                    _free=$(gpu_mem_free_mb "$_g")
                    if [[ "$_free" -eq -1 ]] || [[ "$_free" -ge "$need_mem" ]]; then
                        chosen_gpu="$_g"
                        return 0
                    fi
                fi
            done
        fi

        echo "[SCHED] active=${total_active}/${MAX_CONCURRENT_JOBS}, no GPU slot available; waiting ${SCHED_POLL_SECS}s"
        sleep "$SCHED_POLL_SECS"
    done
}

launch_job() {
    local desc="$1"
    local log_file="$2"
    shift 2

    wait_for_slot
    local assigned_gpu="$chosen_gpu"
    gpu_active_count[$assigned_gpu]=$(( ${gpu_active_count[$assigned_gpu]:-0} + 1 ))
    (
        "$@" --gpu "$assigned_gpu" 2>&1 | tee "$log_file"
    ) &
    local pid=$!
    active_pids+=("$pid")
    active_descs+=("$desc")
    active_logs+=("$log_file")
    active_gpus+=("$assigned_gpu")
    started_jobs=$((started_jobs + 1))
    echo "[LAUNCH] gpu=${assigned_gpu} pid=${pid} active=$(get_active_count)/${MAX_CONCURRENT_JOBS} ${desc}"
}

make_dim_config() {
    local src_cfg="$1"
    local dst_cfg="$2"
    local dim_out="$3"

    "$PYTHON_BIN" - "$src_cfg" "$dst_cfg" "$dim_out" "$TARGET_EPOCH" <<'PY'
import sys
import yaml

src, dst, dim_out, target_epoch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(src, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f)

if "memory" not in conf or not conf["memory"]:
    raise RuntimeError(f"No memory section found in {src}")
if "train" not in conf or not conf["train"]:
    raise RuntimeError(f"No train section found in {src}")

conf["memory"][0]["dim_out"] = dim_out
conf["train"][0]["epoch"] = target_epoch

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(conf, f, sort_keys=False)
PY
}

echo "Logs: $LOG_DIR"
echo "Python: $PYTHON_BIN"
echo "GPUs: ${GPUS[*]}"
echo "Epoch per run: $TARGET_EPOCH"
echo "Repeats per experiment: $REPEATS"
echo "Max concurrent jobs: $MAX_CONCURRENT_JOBS (up to $(( (MAX_CONCURRENT_JOBS + NUM_GPUS - 1) / NUM_GPUS )) per GPU)"
echo "Estimated mem/job: ${EST_MEM_PER_JOB_MB}MB"
echo "Min free mem reserve: ${MIN_FREE_MEM_MB}MB"

for d_idx in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$d_idx]}"
    extra="${EXTRA_ARGS[$d_idx]}"
    extra_args=()
    if [[ -n "$extra" ]]; then
        read -r -a extra_args <<< "$extra"
    fi

    for m_idx in "${!MODELS[@]}"; do
        model="${MODELS[$m_idx]}"
        config="${CONFIGS[$m_idx]}"
        dim_config_base="$REPO_ROOT/$config"

        for dim_out in "${DIM_OUTS[@]}"; do
            dim_config="$TMP_CONFIG_DIR/${model}_dim${dim_out}.yml"
            make_dim_config "$dim_config_base" "$dim_config" "$dim_out"

            for delay in "${DELAYS[@]}"; do
                for run_id in $(seq 1 "$REPEATS"); do
                    log_file="$LOG_DIR/${model}_${dataset}_dim${dim_out}_delay${delay}_run${run_id}_pin.log"

                    echo "============================================================"
                    echo "[${model} / ${dataset} / dim_out=${dim_out} / delay=${delay} / run=${run_id}/${REPEATS} / pin_memory=true / epoch=${TARGET_EPOCH}]"
                    echo "config=$dim_config"
                    echo "log=$log_file"
                    echo "============================================================"

                    launch_job \
                        "${model}/${dataset}/dim${dim_out}/delay${delay}/run${run_id}" \
                        "$log_file" \
                        "$PYTHON_BIN" -u "$REPO_ROOT/train_non_timing.py" \
                        --data "$dataset" \
                        --config "$dim_config" \
                        --model_name "${model}_${dataset}_dim${dim_out}_delay${delay}_run${run_id}_pin" \
                        --pin_memory \
                        --memory_update_delay_batches "$delay" \
                        "${extra_args[@]}"

                    echo ""
                done
            done
        done
    done
done

while [[ "$(get_active_count)" -gt 0 ]]; do
    reap_finished_jobs
    if [[ "$(get_active_count)" -gt 0 ]]; then
        sleep "$SCHED_POLL_SECS"
    fi
done

echo "All runs finished."
echo "Logs saved to $LOG_DIR"
echo "Started jobs: $started_jobs"
echo "Finished jobs: $finished_jobs"

