#!/bin/bash
# Freshness sweep for delayed memory updates.
# Runs train_non_timing.py with pin_memory enabled and sweeps
# memory_update_delay_batches across datasets and model configs.
#
# Usage:
#   bash exp/freshness/freshness.sh [GPU_ID]

set -uo pipefail

GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$SCRIPT_DIR/logs"
SUMMARY_FILE="$LOG_DIR/summary.tsv"
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

cat > "$SUMMARY_FILE" <<'EOF'
model	dataset	dim_out	delay	test_ap	test_metric	test_score	status	log_file
EOF

make_dim_config() {
    local src_cfg="$1"
    local dst_cfg="$2"
    local dim_out="$3"

    "$PYTHON_BIN" - "$src_cfg" "$dst_cfg" "$dim_out" <<'PY'
import sys
import yaml

src, dst, dim_out = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f)

if "memory" not in conf or not conf["memory"]:
    raise RuntimeError(f"No memory section found in {src}")

conf["memory"][0]["dim_out"] = dim_out

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(conf, f, sort_keys=False)
PY
}

append_summary() {
    local model="$1"
    local dataset="$2"
    local dim_out="$3"
    local delay="$4"
    local status="$5"
    local log_file="$6"

    "$PYTHON_BIN" - "$model" "$dataset" "$dim_out" "$delay" "$status" "$log_file" <<'PY' >> "$SUMMARY_FILE"
import pathlib
import re
import sys

model, dataset, dim_out, delay, status, log_file = sys.argv[1:7]
log_path = pathlib.Path(log_file)
test_ap = "NA"
metric_name = "NA"
metric_value = "NA"

if log_path.exists() and status == "ok":
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"test AP:([0-9.eE+-]+)\s+test (AUC|MRR):([0-9.eE+-]+)", text)
    if matches:
        test_ap, metric_name, metric_value = matches[-1]

print(f"{model}\t{dataset}\t{dim_out}\t{delay}\t{test_ap}\t{metric_name}\t{metric_value}\t{status}\t{log_file}")
PY
}

echo "Logs: $LOG_DIR"
echo "Summary: $SUMMARY_FILE"
echo "Python: $PYTHON_BIN"

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
                log_file="$LOG_DIR/${model}_${dataset}_dim${dim_out}_delay${delay}_pin.log"

                echo "============================================================"
                echo "[${model} / ${dataset} / dim_out=${dim_out} / delay=${delay} / pin_memory=true]"
                echo "config=$dim_config"
                echo "log=$log_file"
                echo "============================================================"

                if "$PYTHON_BIN" -u "$REPO_ROOT/train_non_timing.py" \
                    --data "$dataset" \
                    --config "$dim_config" \
                    --gpu "$GPU" \
                    --model_name "${model}_${dataset}_dim${dim_out}_delay${delay}_pin" \
                    --pin_memory \
                    --memory_update_delay_batches "$delay" \
                    "${extra_args[@]}" \
                    2>&1 | tee "$log_file"; then
                    append_summary "$model" "$dataset" "$dim_out" "$delay" "ok" "$log_file"
                else
                    append_summary "$model" "$dataset" "$dim_out" "$delay" "failed" "$log_file"
                fi

                echo ""
            done
        done
    done
done

echo "All runs finished."
echo "Logs saved to $LOG_DIR"
echo "Summary saved to $SUMMARY_FILE"
