#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$SCRIPT_DIR/logs"
TMP_CONFIG_DIR="$LOG_DIR/tmp_configs"
CONFIG_PATH="$SCRIPT_DIR/0_run.yaml"
RUNNER="$REPO_ROOT/accuracy_experiment/run_on_one_gpu.py"
USER_PREFIX="${LOG_USER_PREFIX:-${USER:-$(id -un)}_${HOSTNAME:-$(hostname -s)}}"
ROW_SEP=$'\x1f'

usage() {
    cat <<'EOF'
Usage:
  bash accuracy_experiment/freshness/0_run.sh [GPU_IDS]
  uv run python accuracy_experiment/run_on_one_gpu.py --script accuracy_experiment/freshness/0_run.sh [--gpus 0,1]

This script defines freshness experiment jobs.
When called normally, it delegates scheduling to accuracy_experiment/run_on_one_gpu.py.
When called with --emit-jobs, it prints job definitions for the scheduler.
Experiment settings are loaded from accuracy_experiment/freshness/0_run.yaml.
Use `seeds:` in the YAML to enumerate exact seeds for each run.
Use `batch_sizes:` in the YAML to enumerate exact batch sizes for each run.
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
        runner_args=(--script "$0" --dataset "$DATASET")
        if [[ "${#FORWARD_ARGS[@]}" -gt 0 ]]; then
            runner_args+=(--gpus "${FORWARD_ARGS[0]}")
            if [[ "${#FORWARD_ARGS[@]}" -gt 1 ]]; then
                runner_args+=(-- "${FORWARD_ARGS[@]:1}")
            fi
        fi
        echo "Running for dataset: $DATASET"
        "$RUNNER" "${runner_args[@]}"
    done
    exit 0
fi

PYTHON_BIN="$(resolve_python_bin)"

mkdir -p "$LOG_DIR" "$TMP_CONFIG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file not found: $CONFIG_PATH" >&2
    exit 1
fi

make_dim_config() {
    local src_cfg="$1"
    local dst_cfg="$2"
    local dim_out="$3"
    local batch_size="$4"
    local target_epoch="$5"

    "$PYTHON_BIN" - "$src_cfg" "$dst_cfg" "$dim_out" "$batch_size" "$target_epoch" <<'PY'
import sys
import yaml

src, dst, dim_out, batch_size, target_epoch = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    int(sys.argv[4]),
    int(sys.argv[5]),
)
with open(src, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f)

if "memory" not in conf or not conf["memory"]:
    raise RuntimeError(f"No memory section found in {src}")
if "train" not in conf or not conf["train"]:
    raise RuntimeError(f"No train section found in {src}")

conf["memory"][0]["dim_out"] = dim_out
conf["train"][0]["batch_size"] = batch_size
conf["train"][0]["epoch"] = target_epoch

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(conf, f, sort_keys=False)
PY
}

get_train_meta() {
    local cfg="$1"

    "$PYTHON_BIN" - "$cfg" <<'PY'
import sys
import yaml

cfg = sys.argv[1]
with open(cfg, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f)

if "train" not in conf or not conf["train"]:
    raise RuntimeError(f"No train section found in {cfg}")

train = conf["train"][0]
print(f"{int(train['batch_size'])}\t{int(train['epoch'])}")
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
delays = conf.get("delays") or []
dim_outs = conf.get("dim_outs") or []
batch_sizes = conf.get("batch_sizes")
single_batch_size = conf.get("batch_size")
target_epoch = int(conf.get("target_epoch", 1))
seed_values = conf.get("seeds")

if not models:
    raise RuntimeError("0_run.yaml must define at least one model")
if not datasets:
    raise RuntimeError("0_run.yaml must define at least one dataset")
if not delays:
    raise RuntimeError("0_run.yaml must define at least one delay")
if not dim_outs:
    raise RuntimeError("0_run.yaml must define at least one dim_out")
if batch_sizes is not None and single_batch_size is not None:
    raise RuntimeError("0_run.yaml should define only one of `batch_size` or `batch_sizes`")

if batch_sizes is None:
    if single_batch_size is None:
        batch_sizes = [600]
    else:
        batch_sizes = [int(single_batch_size)]
else:
    if not isinstance(batch_sizes, list) or not batch_sizes:
        raise RuntimeError("0_run.yaml `batch_sizes` must be a non-empty list when provided")
    batch_sizes = [int(batch_size) for batch_size in batch_sizes]

if seed_values is None:
    repeats = int(conf.get("repeats", 1))
    seeds = list(range(1, repeats + 1))
else:
    if not isinstance(seed_values, list) or not seed_values:
        raise RuntimeError("0_run.yaml `seeds` must be a non-empty list when provided")
    seeds = [int(seed) for seed in seed_values]

seed_count = len(seeds)

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
        for dim_out in dim_outs:
            for batch_size in batch_sizes:
                for delay in delays:
                    for seed in seeds:
                        print(
                            "\x1f".join(
                                [
                                    model_name,
                                    model_config,
                                    dataset_name,
                                    extra_args,
                                    str(dim_out),
                                    str(delay),
                                    str(seed),
                                    str(batch_size),
                                    str(seed_count),
                                    str(target_epoch),
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
IFS="$ROW_SEP" read -r _ _ _ _ _ _ _ _ SEED_COUNT TARGET_EPOCH <<< "$first_row"
declare -A seen_batch_sizes
batch_size_list=()
for row in "${experiment_rows[@]}"; do
    IFS="$ROW_SEP" read -r _ _ _ _ _ _ _ batch_size _ _ <<< "$row"
    if [[ -z "${seen_batch_sizes[$batch_size]:-}" ]]; then
        seen_batch_sizes[$batch_size]=1
        batch_size_list+=("$batch_size")
    fi
done
batch_size_summary="$(IFS=,; echo "${batch_size_list[*]}")"

echo "Logs: $LOG_DIR" >&2
echo "Python: $PYTHON_BIN" >&2
echo "Config: $CONFIG_PATH" >&2
echo "Batch sizes per run: $batch_size_summary" >&2
echo "Epoch per run: $TARGET_EPOCH" >&2
echo "Seeds per experiment: $SEED_COUNT" >&2

declare -A prepared_dim_configs

for row in "${experiment_rows[@]}"; do
    IFS="$ROW_SEP" read -r model config dataset extra dim_out delay seed batch_size seed_count target_epoch <<< "$row"
    extra_args=()
    if [[ -n "$extra" ]]; then
        read -r -a extra_args <<< "$extra"
    fi

    dim_config_key="${model}:${dim_out}:${batch_size}:${target_epoch}"
    dim_config="$TMP_CONFIG_DIR/${model}_dim${dim_out}_bs${batch_size}_ep${target_epoch}.yml"
    if [[ -z "${prepared_dim_configs[$dim_config_key]:-}" ]]; then
        make_dim_config "$REPO_ROOT/$config" "$dim_config" "$dim_out" "$batch_size" "$target_epoch"
        prepared_dim_configs[$dim_config_key]=1
    fi
    IFS=$'\t' read -r batch_size_cfg epoch_cfg < <(get_train_meta "$dim_config")

    log_file="$LOG_DIR/${USER_PREFIX}_${model}_${dataset}_bs${batch_size_cfg}_memdim${dim_out}_ep${epoch_cfg}_delay${delay}_run${seed}_pin.log"
    desc="${model}/${dataset}/bs${batch_size_cfg}/memdim${dim_out}/ep${epoch_cfg}/delay${delay}/seed${seed}"
    cmd=(
        "$PYTHON_BIN" -u "$REPO_ROOT/train_non_timing_on_gpu.py"
        --data "$dataset"
        --config "$dim_config"
        --model_name "${model}_${dataset}_dim${dim_out}_delay${delay}_seed${seed}_pin"
        --seed "$seed"
        --pin_memory
        --memory_update_delay_batches "$delay"
        --gpu "__RUN_ON_ONE_GPU_ASSIGNED_GPU__"
    )
    if [[ "${#extra_args[@]}" -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi

    echo "============================================================" >&2
    echo "[${model} / ${dataset} / batch_size=${batch_size_cfg} / dim_out=${dim_out} / epoch=${epoch_cfg} / delay=${delay} / seed=${seed} / total_seeds=${seed_count} / pin_memory=true]" >&2
    echo "config=$dim_config" >&2
    echo "log=$log_file" >&2
    echo "============================================================" >&2

    printf '%s\t%s\t' "$desc" "$log_file"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    echo "" >&2
done
