#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RUNNER="$REPO_ROOT/accuracy_experiment/run_on_gpu_pairs.py"
CONFIG_PATH="$SCRIPT_DIR/0_run.yaml"
ROW_SEP=$'\x1f'

usage() {
    cat <<'EOF'
Usage:
  bash accuracy_experiment/compare_ap_tgl_vs_frost/0_run.sh [GPU_IDS]
  python accuracy_experiment/run_on_gpu_pairs.py --script accuracy_experiment/compare_ap_tgl_vs_frost/0_run.sh [--gpus 0,1,2,3]

This script defines compare_ap_tgl_vs_frost jobs.
When called normally, it delegates scheduling to accuracy_experiment/run_on_gpu_pairs.py.
When called with --emit-jobs, it prints job definitions for the scheduler.
Experiment settings are loaded from accuracy_experiment/compare_ap_tgl_vs_frost/0_run.yaml.
Each emitted job always uses exactly 2 GPUs.
EOF
}

resolve_python_bin() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        printf '%s\n' "$PYTHON_BIN"
    elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
        printf '%s\n' "${REPO_ROOT}/.venv/bin/python"
    elif command -v python >/dev/null 2>&1; then
        command -v python
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        echo "Python interpreter not found." >&2
        exit 1
    fi
}

load_dataset_names() {
    local config_path="$1"
    local dataset_filter="${2:-}"

    "$PYTHON_BIN" - "$config_path" "$dataset_filter" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
dataset_filter = sys.argv[2] or None

with open(config_path, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f) or {}

datasets = conf.get("datasets") or []
if not isinstance(datasets, list):
    raise RuntimeError("0_run.yaml `datasets` must be a list")

matched = 0
for dataset in datasets:
    if isinstance(dataset, str):
        dataset_name = dataset
    elif isinstance(dataset, dict):
        dataset_name = dataset.get("name")
    else:
        raise RuntimeError("Each dataset entry must be either a string or a mapping with `name`")

    if not dataset_name:
        raise RuntimeError("Each dataset entry must define a non-empty dataset name")
    if dataset_filter and dataset_name != dataset_filter:
        continue

    print(dataset_name)
    matched += 1

if dataset_filter and matched == 0:
    raise RuntimeError(f"Dataset not found in {config_path}: {dataset_filter}")
PY
}

EMIT_JOBS=0
DATASET_FILTER=""
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --emit-jobs)
            EMIT_JOBS=1
            shift
            ;;
        --dataset)
            if [[ $# -lt 2 ]]; then
                echo "--dataset requires a value" >&2
                exit 1
            fi
            DATASET_FILTER="$2"
            shift 2
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

    mapfile -t DATASETS < <(load_dataset_names "$CONFIG_PATH" "$DATASET_FILTER")
    if [[ ${#DATASETS[@]} -eq 0 ]]; then
        echo "No datasets found in $CONFIG_PATH" >&2
        exit 1
    fi

    for DATASET in "${DATASETS[@]}"; do
        runner_args=("$RUNNER" --script "$0" --dataset "$DATASET")
        if [[ "${#FORWARD_ARGS[@]}" -gt 0 ]]; then
            runner_args+=(--gpus "${FORWARD_ARGS[0]}")
            if [[ "${#FORWARD_ARGS[@]}" -gt 1 ]]; then
                runner_args+=(-- "${FORWARD_ARGS[@]:1}")
            fi
        fi
        echo "Running for dataset: $DATASET"
        "$PYTHON_BIN" "${runner_args[@]}"
    done
    exit 0
fi

PYTHON_BIN="$(resolve_python_bin)"
if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
fi

OMP_THREADS="${OMP_NUM_THREADS:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29750}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
TASK_NUM_GPUS=2
TORCH_PROC_PER_NODE=$((TASK_NUM_GPUS + 1))
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
TMP_CONFIG_DIR="$SCRIPT_DIR/tmp_configs/$RUN_TAG"
USER_PREFIX="${LOG_USER_PREFIX:-${USER:-$(id -un)}_${HOSTNAME:-$(hostname -s)}}"

if [[ -n "${NUM_GPUS:-}" && "${NUM_GPUS}" != "${TASK_NUM_GPUS}" ]]; then
    echo "NUM_GPUS is fixed to ${TASK_NUM_GPUS} for this script, got: ${NUM_GPUS}" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$TMP_CONFIG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file not found: $CONFIG_PATH" >&2
    exit 1
fi

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

load_experiment_rows() {
    "$PYTHON_BIN" - "$CONFIG_PATH" "$DATASET_FILTER" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
dataset_filter = sys.argv[2] or None
with open(config_path, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f) or {}

models = conf.get("models") or []
datasets = conf.get("datasets") or []
mem_dims = conf.get("mem_dims") or []
seeds = conf.get("seeds") or [0]
target_epoch = int(conf.get("target_epoch", 1))
raw_batch_sizes = conf.get("batch_sizes")

if raw_batch_sizes is None:
    batch_sizes = [int(conf.get("batch_size", 1))]
else:
    batch_sizes = [int(batch_size) for batch_size in raw_batch_sizes]

if not models:
    raise RuntimeError("0_run.yaml must define at least one model")
if not datasets:
    raise RuntimeError("0_run.yaml must define at least one dataset")
if not mem_dims:
    raise RuntimeError("0_run.yaml must define at least one mem_dim")
if not batch_sizes:
    raise RuntimeError("0_run.yaml must define at least one batch_size or batch_sizes entry")

matched_datasets = 0
for dataset in datasets:
    if isinstance(dataset, str):
        dataset_name = dataset
        extra_args = ""
    elif isinstance(dataset, dict):
        dataset_name = dataset.get("name")
        if not dataset_name:
            raise RuntimeError("Each dataset mapping must define a non-empty `name`")
        extra_args = dataset.get("extra_args", "")
    else:
        raise RuntimeError("Each dataset entry must be either a string or a mapping with `name`")

    if extra_args is None:
        extra_args = ""
    elif isinstance(extra_args, list):
        extra_args = " ".join(str(arg) for arg in extra_args)
    else:
        extra_args = str(extra_args)

    if dataset_filter and dataset_name != dataset_filter:
        continue

    matched_datasets += 1
    for model in models:
        model_name = model["name"]
        model_config = model["config"]
        for mem_dim in mem_dims:
            for batch_size in batch_sizes:
                for seed in seeds:
                    print(
                        "\x1f".join(
                            [
                                model_name,
                                model_config,
                                dataset_name,
                                extra_args,
                                str(mem_dim),
                                str(seed),
                                str(target_epoch),
                                str(batch_size),
                            ]
                        )
                    )

if dataset_filter and matched_datasets == 0:
    raise RuntimeError(f"Dataset not found in {config_path}: {dataset_filter}")
PY
}

mapfile -t experiment_rows < <(load_experiment_rows)

if [[ "${#experiment_rows[@]}" -eq 0 ]]; then
    echo "No experiments generated from $CONFIG_PATH" >&2
    exit 1
fi

first_row="${experiment_rows[0]}"
IFS="$ROW_SEP" read -r _ _ _ _ _ _ TARGET_EPOCH _first_batch_size <<< "$first_row"

declare -A seen_batch_sizes
batch_sizes=()
for row in "${experiment_rows[@]}"; do
    IFS="$ROW_SEP" read -r _ _ _ _ _ _ _ row_batch_size <<< "$row"
    if [[ -z "${seen_batch_sizes[$row_batch_size]:-}" ]]; then
        batch_sizes+=("$row_batch_size")
        seen_batch_sizes[$row_batch_size]=1
    fi
done

echo "Logs: $LOG_DIR" >&2
echo "Tmp configs: $TMP_CONFIG_DIR" >&2
echo "Python: $PYTHON_BIN" >&2
echo "Config: $CONFIG_PATH" >&2
echo "Task GPUs per job: $TASK_NUM_GPUS" >&2
echo "Batch sizes: ${batch_sizes[*]}" >&2
echo "Epochs: $TARGET_EPOCH" >&2
echo "OMP threads: $OMP_THREADS" >&2
echo "Total runs: ${#experiment_rows[@]}" >&2

declare -A prepared_batch_configs
run_idx=0

for row in "${experiment_rows[@]}"; do
    IFS="$ROW_SEP" read -r model config dataset extra mem_dim seed target_epoch batch_size <<< "$row"

    if [[ ! -f "$REPO_ROOT/$config" ]]; then
        echo "Config file not found: $REPO_ROOT/$config" >&2
        exit 1
    fi

    extra_args=()
    if [[ -n "$extra" ]]; then
        read -r -a extra_args <<< "$extra"
    fi

    batch_config="$TMP_CONFIG_DIR/${model}_${dataset}_memdim${mem_dim}_ep${target_epoch}_bs${batch_size}.yml"
    batch_config_key="$batch_config"
    if [[ -z "${prepared_batch_configs[$batch_config_key]:-}" ]]; then
        make_experiment_config "$REPO_ROOT/$config" "$batch_config" "$batch_size" "$target_epoch" "$mem_dim"
        prepared_batch_configs[$batch_config_key]=1
    fi

    master_port=$((BASE_MASTER_PORT + run_idx))
    log_file="$LOG_DIR/${USER_PREFIX}_${model}_${dataset}_bs${batch_size}_ngpu${TASK_NUM_GPUS}_memdim${mem_dim}_ep${target_epoch}_seed${seed}.log"
    desc="${model}/${dataset}/bs${batch_size}/memdim${mem_dim}/ep${target_epoch}/seed${seed}"

    cmd=(
        "$PYTHON_BIN" -u -m torch.distributed.run
        --nproc_per_node "$TORCH_PROC_PER_NODE"
        --master_addr "$MASTER_ADDR"
        --master_port "$master_port"
        "$REPO_ROOT/train_dist.py"
        --seed "$seed"
        --dataset "$dataset"
        --config "$batch_config"
        --num_gpus "$TASK_NUM_GPUS"
        --omp_num_threads "$OMP_THREADS"
    )

    if [[ "${#extra_args[@]}" -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi

    echo "============================================================" >&2
    echo "[run $((run_idx + 1))/${#experiment_rows[@]}] model=${model} dataset=${dataset} batch_size=${batch_size} mem_dim=${mem_dim} epoch=${target_epoch} seed=${seed}" >&2
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
