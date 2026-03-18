#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$SCRIPT_DIR/logs"
TMP_CONFIG_DIR="$LOG_DIR/tmp_configs"
CONFIG_PATH="$SCRIPT_DIR/0_run.yaml"
RUNNER="$REPO_ROOT/exp/run_on_one_gpu.py"

usage() {
    cat <<'EOF'
Usage:
  bash exp/freshness/0_run.sh [GPU_IDS]
  python exp/run_on_one_gpu.py --script exp/freshness/0_run.sh [--gpus 0,1]

This script defines freshness experiment jobs.
When called normally, it delegates scheduling to exp/run_on_one_gpu.py.
When called with --emit-jobs, it prints job definitions for the scheduler.
Experiment settings are loaded from exp/freshness/0_run.yaml.
EOF
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
    runner_args=(--script "$0")
    if [[ "${#FORWARD_ARGS[@]}" -gt 0 ]]; then
        runner_args+=(--gpus "${FORWARD_ARGS[0]}")
        if [[ "${#FORWARD_ARGS[@]}" -gt 1 ]]; then
            runner_args+=(-- "${FORWARD_ARGS[@]:1}")
        fi
    fi
    exec "$RUNNER" "${runner_args[@]}"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="/home/sqp17/miniconda3/envs/simple_py310/bin/python"
fi

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
    "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f) or {}

models = conf.get("models") or []
datasets = conf.get("datasets") or []
delays = conf.get("delays") or []
dim_outs = conf.get("dim_outs") or []
repeats = int(conf.get("repeats", 1))
target_epoch = int(conf.get("target_epoch", 1))

if not models:
    raise RuntimeError("0_run.yaml must define at least one model")
if not datasets:
    raise RuntimeError("0_run.yaml must define at least one dataset")
if not delays:
    raise RuntimeError("0_run.yaml must define at least one delay")
if not dim_outs:
    raise RuntimeError("0_run.yaml must define at least one dim_out")

for dataset in datasets:
    dataset_name = dataset["name"]
    extra_args = dataset.get("extra_args", "")
    for model in models:
        model_name = model["name"]
        model_config = model["config"]
        for dim_out in dim_outs:
            for delay in delays:
                for run_id in range(1, repeats + 1):
                    print(
                        "\t".join(
                            [
                                model_name,
                                model_config,
                                dataset_name,
                                extra_args,
                                str(dim_out),
                                str(delay),
                                str(run_id),
                                str(repeats),
                                str(target_epoch),
                            ]
                        )
                    )
PY
}

mapfile -t experiment_rows < <(load_experiment_rows)

if [[ "${#experiment_rows[@]}" -eq 0 ]]; then
    echo "No experiments generated from $CONFIG_PATH" >&2
    exit 1
fi

first_row="${experiment_rows[0]}"
IFS=$'\t' read -r _ _ _ _ _ _ _ REPEATS TARGET_EPOCH <<< "$first_row"

echo "Logs: $LOG_DIR" >&2
echo "Python: $PYTHON_BIN" >&2
echo "Config: $CONFIG_PATH" >&2
echo "Epoch per run: $TARGET_EPOCH" >&2
echo "Repeats per experiment: $REPEATS" >&2

declare -A prepared_dim_configs

for row in "${experiment_rows[@]}"; do
    IFS=$'\t' read -r model config dataset extra dim_out delay run_id repeats target_epoch <<< "$row"
    extra_args=()
    if [[ -n "$extra" ]]; then
        read -r -a extra_args <<< "$extra"
    fi

    dim_config_key="${model}:${dim_out}"
    dim_config="$TMP_CONFIG_DIR/${model}_dim${dim_out}.yml"
    if [[ -z "${prepared_dim_configs[$dim_config_key]:-}" ]]; then
        make_dim_config "$REPO_ROOT/$config" "$dim_config" "$dim_out"
        prepared_dim_configs[$dim_config_key]=1
    fi
    IFS=$'\t' read -r batch_size_cfg epoch_cfg < <(get_train_meta "$dim_config")

    log_file="$LOG_DIR/${model}_${dataset}_bs${batch_size_cfg}_memdim${dim_out}_ep${epoch_cfg}_delay${delay}_run${run_id}_pin.log"
    desc="${model}/${dataset}/bs${batch_size_cfg}/memdim${dim_out}/ep${epoch_cfg}/delay${delay}/run${run_id}"
    cmd=(
        "$PYTHON_BIN" -u "$REPO_ROOT/train_non_timing.py"
        --data "$dataset"
        --config "$dim_config"
        --model_name "${model}_${dataset}_dim${dim_out}_delay${delay}_run${run_id}_pin"
        --pin_memory
        --memory_update_delay_batches "$delay"
        --gpu "__RUN_ON_ONE_GPU_ASSIGNED_GPU__"
    )
    if [[ "${#extra_args[@]}" -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi

    echo "============================================================" >&2
    echo "[${model} / ${dataset} / batch_size=${batch_size_cfg} / dim_out=${dim_out} / epoch=${epoch_cfg} / delay=${delay} / run=${run_id}/${repeats} / pin_memory=true]" >&2
    echo "config=$dim_config" >&2
    echo "log=$log_file" >&2
    echo "============================================================" >&2

    printf '%s\t%s\t' "$desc" "$log_file"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    echo "" >&2
done
