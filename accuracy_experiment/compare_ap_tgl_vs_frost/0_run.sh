#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RUNNER="$REPO_ROOT/experiments/run_on_gpu_pairs.py"

usage() {
    cat <<'EOF'
Usage:
  bash experiments/compare_ap_tgl_vs_frost/0_run.sh [GPU_IDS]
  python experiments/run_on_gpu_pairs.py --script experiments/compare_ap_tgl_vs_frost/0_run.sh [--gpus 0,1,2,3]

This script defines compare_ap_tgl_vs_frost jobs.
When called normally, it delegates scheduling to experiments/run_on_gpu_pairs.py.
When called with --emit-jobs, it prints job definitions for the scheduler.
Each emitted job always uses exactly 2 GPUs.
EOF
}

resolve_python_bin() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        printf '%s\n' "$PYTHON_BIN"
    elif command -v python >/dev/null 2>&1; then
        command -v python
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        echo "Python interpreter not found." >&2
        exit 1
    fi
}

EMIT_JOBS=0
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --emit-jobs)
            EMIT_JOBS=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$EMIT_JOBS" -eq 0 ]]; then
    PYTHON_BIN="$(resolve_python_bin)"
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v "$PYTHON_BIN")"
    fi

    runner_args=("$RUNNER" --script "$0")
    if [[ "${#FORWARD_ARGS[@]}" -gt 0 ]]; then
        runner_args+=(--gpus "${FORWARD_ARGS[0]}")
        if [[ "${#FORWARD_ARGS[@]}" -gt 1 ]]; then
            runner_args+=(-- "${FORWARD_ARGS[@]:1}")
        fi
    fi

    exec "$PYTHON_BIN" "${runner_args[@]}"
fi

PYTHON_BIN="$(resolve_python_bin)"
if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
fi

OMP_THREADS="${OMP_NUM_THREADS:-8}"
MODEL="${MODEL:-TGN}"
DATASET="${DATASET:-LASTFM}"
EPOCHS="${EPOCHS:-100}"
REPEATS="${REPEATS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4000}"
MEM_DIMS_CSV="${MEM_DIMS:-128,256,512}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29750}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
TASK_NUM_GPUS=2
TORCH_PROC_PER_NODE=$((TASK_NUM_GPUS + 1))

if [[ -n "${NUM_GPUS:-}" && "${NUM_GPUS}" != "${TASK_NUM_GPUS}" ]]; then
    echo "NUM_GPUS is fixed to ${TASK_NUM_GPUS} for this script, got: ${NUM_GPUS}" >&2
    exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/${MODEL}.yml}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
TMP_CONFIG_DIR="$SCRIPT_DIR/tmp_configs/$RUN_TAG"

mkdir -p "$LOG_DIR" "$TMP_CONFIG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file not found: $CONFIG_PATH" >&2
    exit 1
fi

IFS=',' read -r -a MEM_DIMS <<< "$MEM_DIMS_CSV"

get_dataset_extra_args() {
    local dataset="$1"
    case "$dataset" in
        LASTFM|MOOC)
            printf '%s\n' '--rnd_edim 128'
            ;;
        *)
            printf '\n'
            ;;
    esac
}

make_experiment_config() {
    local src_cfg="$1"
    local dst_cfg="$2"
    local batch_size="$3"
    local epochs="$4"
    local mem_dim="$5"

    "$PYTHON_BIN" - "$src_cfg" "$dst_cfg" "$batch_size" "$epochs" "$mem_dim" <<'PY'
import sys
import yaml

src, dst, batch_size, epochs, mem_dim = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
with open(src, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f)

if "train" not in conf or not conf["train"]:
    raise RuntimeError(f"No train section found in {src}")
if "memory" not in conf or not conf["memory"]:
    raise RuntimeError(f"No memory section found in {src}")

conf["train"][0]["batch_size"] = batch_size
conf["train"][0]["epoch"] = epochs
conf["memory"][0]["dim_out"] = mem_dim

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(conf, f, sort_keys=False)
PY
}

extra_args="$(get_dataset_extra_args "$DATASET")"
total_runs=$((REPEATS * ${#MEM_DIMS[@]}))
run_idx=0

echo "Logs: $LOG_DIR" >&2
echo "Tmp configs: $TMP_CONFIG_DIR" >&2
echo "Python: $PYTHON_BIN" >&2
echo "Model: $MODEL" >&2
echo "Dataset: $DATASET" >&2
echo "Task GPUs per job: $TASK_NUM_GPUS" >&2
echo "Batch size: $BATCH_SIZE" >&2
echo "Epochs: $EPOCHS" >&2
echo "Repeats: $REPEATS" >&2
echo "Mem dims: ${MEM_DIMS[*]}" >&2
echo "OMP threads: $OMP_THREADS" >&2
echo "Total runs: $total_runs" >&2

for raw_mem_dim in "${MEM_DIMS[@]}"; do
    mem_dim="${raw_mem_dim//[[:space:]]/}"
    batch_config="$TMP_CONFIG_DIR/${MODEL}_${DATASET}_memdim${mem_dim}_ep${EPOCHS}_bs${BATCH_SIZE}.yml"
    make_experiment_config "$CONFIG_PATH" "$batch_config" "$BATCH_SIZE" "$EPOCHS" "$mem_dim"

    for ((repeat_idx = 1; repeat_idx <= REPEATS; repeat_idx++)); do
        master_port=$((BASE_MASTER_PORT + run_idx))
        log_file="$LOG_DIR/${MODEL}_${DATASET}_ngpu${TASK_NUM_GPUS}_memdim${mem_dim}_ep${EPOCHS}_rep${repeat_idx}.log"
        desc="${MODEL}/${DATASET}/memdim${mem_dim}/rep${repeat_idx}"

        cmd=(
            "$PYTHON_BIN" -u -m torch.distributed.run
            --nproc_per_node "$TORCH_PROC_PER_NODE"
            --master_addr "$MASTER_ADDR"
            --master_port "$master_port"
            "$REPO_ROOT/train_dist.py"
            --dataset "$DATASET"
            --config "$batch_config"
            --num_gpus "$TASK_NUM_GPUS"
            --omp_num_threads "$OMP_THREADS"
        )

        if [[ -n "$extra_args" ]]; then
            read -r -a extra_parts <<< "$extra_args"
            cmd+=("${extra_parts[@]}")
        fi

        echo "============================================================" >&2
        echo "[run $((run_idx + 1))/$total_runs] model=${MODEL} dataset=${DATASET} mem_dim=${mem_dim} repeat=${repeat_idx}/${REPEATS}" >&2
        echo "config: ${batch_config}" >&2
        echo "log: ${log_file}" >&2
        echo "master_port: ${master_port}" >&2
        echo "============================================================" >&2

        printf '%s\t%s\t' "$desc" "$log_file"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        echo "" >&2

        run_idx=$((run_idx + 1))
    done
done
