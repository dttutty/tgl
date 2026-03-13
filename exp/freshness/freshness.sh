#!/bin/bash
# Freshness sweep for delayed memory updates.
# Runs train_non_timing.py with pin_memory enabled and sweeps
# memory_update_delay_batches across datasets and model configs.
#
# Usage:
#   bash exp/freshness/freshness.sh [GPU_ID]
#
# Optional env vars for concurrent scheduling:
#   MAX_CONCURRENT_JOBS (default: 4)
#   EST_MEM_PER_JOB_MB (default: 5000)
#   MIN_FREE_MEM_MB (default: 1500)
#   SCHED_POLL_SECS (default: 10)

set -uo pipefail

GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$SCRIPT_DIR/logs"
PYTHON_BIN="${PYTHON_BIN:-/home/sqp17/miniconda3/envs/simple_py310/bin/python}"
TMP_CONFIG_DIR="$LOG_DIR/tmp_configs"

mkdir -p "$LOG_DIR"
mkdir -p "$TMP_CONFIG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
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
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-4}"
EST_MEM_PER_JOB_MB="${EST_MEM_PER_JOB_MB:-5000}"
MIN_FREE_MEM_MB="${MIN_FREE_MEM_MB:-1500}"
SCHED_POLL_SECS="${SCHED_POLL_SECS:-10}"

active_pids=()
active_descs=()
active_logs=()
started_jobs=0
finished_jobs=0

get_active_count() {
    echo "${#active_pids[@]}"
}

gpu_mem_free_mb() {
    local line
    local total
    local used
    if ! line=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null); then
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
    local i
    local pid
    local desc
    local logf

    for i in "${!active_pids[@]}"; do
        pid="${active_pids[$i]}"
        desc="${active_descs[$i]}"
        logf="${active_logs[$i]}"
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
            new_descs+=("$desc")
            new_logs+=("$logf")
        else
            if wait "$pid"; then
                echo "[DONE] ${desc}"
            else
                echo "[FAILED] ${desc} (see ${logf})" >&2
            fi
            finished_jobs=$((finished_jobs + 1))
        fi
    done

    active_pids=("${new_pids[@]}")
    active_descs=("${new_descs[@]}")
    active_logs=("${new_logs[@]}")
}

wait_for_slot() {
    local free_mem
    local need_mem
    local active
    need_mem=$((EST_MEM_PER_JOB_MB + MIN_FREE_MEM_MB))

    while true; do
        reap_finished_jobs
        active=$(get_active_count)
        free_mem=$(gpu_mem_free_mb)

        if [[ "$active" -lt "$MAX_CONCURRENT_JOBS" ]]; then
            if [[ "$free_mem" -eq -1 ]]; then
                # nvidia-smi unavailable: fall back to concurrency cap only.
                break
            fi
            if [[ "$free_mem" -ge "$need_mem" ]]; then
                break
            fi
        fi

        echo "[SCHED] active=${active}/${MAX_CONCURRENT_JOBS}, free_mem=${free_mem}MB, need>=${need_mem}MB; waiting ${SCHED_POLL_SECS}s"
        sleep "$SCHED_POLL_SECS"
    done
}

launch_job() {
    local desc="$1"
    local log_file="$2"
    shift 2

    wait_for_slot
    (
        "$@" 2>&1 | tee "$log_file"
    ) &
    local pid=$!
    active_pids+=("$pid")
    active_descs+=("$desc")
    active_logs+=("$log_file")
    started_jobs=$((started_jobs + 1))
    echo "[LAUNCH] pid=${pid} active=$(get_active_count)/${MAX_CONCURRENT_JOBS} ${desc}"
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
echo "Epoch per run: $TARGET_EPOCH"
echo "Repeats per experiment: $REPEATS"
echo "Max concurrent jobs: $MAX_CONCURRENT_JOBS"
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
                        --gpu "$GPU" \
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

