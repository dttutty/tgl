#!/usr/bin/env bash
set -euo pipefail

# Validated on this machine for NCCL-only DDP:
# - Python 3.10
# - torch 2.10.0+cu128
# - torch-scatter 2.1.2+pt210cu128
# - torchdata 0.9.0
# - CUDA-enabled DGL copied from an existing working env
#
# Why copy DGL instead of pip-installing it here?
# DGL does not currently publish a public torch-2.10/cu128 wheel repo, while
# the older torch 2.7.0 stack reproduced the NCCL illegal memory access bug on
# this Blackwell workstation. The combination below was validated locally and
# gets past the old crash point with NCCL.

ENV_DIR="${ENV_DIR:-/tmp/tgl_t210_nccl}"
SRC_DGL_ENV="${SRC_DGL_ENV:-/home/sqp17/miniforge3/envs/simple_py310}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "${SRC_DGL_ENV}/bin/python" ]]; then
  echo "Source DGL environment not found: ${SRC_DGL_ENV}" >&2
  echo "Set SRC_DGL_ENV to an env that already contains CUDA-enabled dgl." >&2
  exit 1
fi

# Reuse the source env's Python ABI so the copied DGL wheel stays compatible.
BOOTSTRAP_PYTHON="${SRC_DGL_ENV}/bin/python"

echo "Creating venv at ${ENV_DIR}"
"${BOOTSTRAP_PYTHON}" -m venv "${ENV_DIR}"

PYTHON_BIN="${ENV_DIR}/bin/python"
PIP_BIN=("${PYTHON_BIN}" -m pip)

echo "Upgrading pip/setuptools/wheel"
"${PIP_BIN[@]}" install --upgrade pip setuptools wheel

echo "Installing torch 2.10.0 + CUDA 12.8"
"${PIP_BIN[@]}" install \
  torch==2.10.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

echo "Installing Python dependencies"
"${PIP_BIN[@]}" install \
  numpy \
  pandas \
  packaging \
  psutil \
  pybind11 \
  pydantic \
  pyyaml \
  requests \
  scikit-learn \
  scipy \
  torch-scatter \
  torchdata==0.9.0 \
  tqdm \
  -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

SRC_SITE_PACKAGES="$("${SRC_DGL_ENV}/bin/python" - <<'PY'
import site
paths = [p for p in site.getsitepackages() if p.endswith("site-packages")]
if not paths:
    raise SystemExit("Could not resolve source site-packages")
print(paths[0])
PY
)"

DST_SITE_PACKAGES="$("${PYTHON_BIN}" - <<'PY'
import site
paths = [p for p in site.getsitepackages() if p.endswith("site-packages")]
if not paths:
    raise SystemExit("Could not resolve destination site-packages")
print(paths[0])
PY
)"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        "This script expects Python 3.10 because the copied DGL package is cp310."
    )
PY

SRC_DGL_DIR="${SRC_SITE_PACKAGES}/dgl"
SRC_DGL_DIST_INFO="$(find "${SRC_SITE_PACKAGES}" -maxdepth 1 -type d -name 'dgl-*.dist-info' | head -n 1)"

if [[ ! -d "${SRC_DGL_DIR}" ]]; then
  echo "Could not find ${SRC_DGL_DIR}" >&2
  exit 1
fi

if [[ -z "${SRC_DGL_DIST_INFO}" ]]; then
  echo "Could not find dgl-*.dist-info under ${SRC_SITE_PACKAGES}" >&2
  exit 1
fi

echo "Copying CUDA-enabled DGL from ${SRC_DGL_ENV}"
rm -rf "${DST_SITE_PACKAGES}/dgl"
find "${DST_SITE_PACKAGES}" -maxdepth 1 -type d -name 'dgl-*.dist-info' -exec rm -rf {} +
cp -a "${SRC_DGL_DIR}" "${DST_SITE_PACKAGES}/"
cp -a "${SRC_DGL_DIST_INFO}" "${DST_SITE_PACKAGES}/"

echo "Building sampler_core extension in-place"
(
  cd "${REPO_DIR}"
  "${PYTHON_BIN}" setup.py build_ext --inplace
)

echo "Smoke test"
"${PYTHON_BIN}" - <<'PY'
import torch
import dgl
import torch_scatter
import torchdata

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("dgl:", dgl.__version__)
print("torch_scatter:", torch_scatter.__version__)
print("torchdata:", torchdata.__version__)
PY

cat <<EOF

Environment is ready.

Activate:
  source "${ENV_DIR}/bin/activate"

Run:
  CUDA_VISIBLE_DEVICES=0,1 \\
  CUDA_LAUNCH_BLOCKING=1 \\
  TORCH_SHOW_CPP_STACKTRACES=1 \\
  TORCH_DISTRIBUTED_DEBUG=DETAIL \\
  NCCL_DEBUG=INFO \\
  "${PYTHON_BIN}" -m torch.distributed.run \\
    --nproc_per_node=3 \\
    --master_addr=127.0.0.1 \\
    --master_port=29730 \\
    train_dist.py --dataset LASTFM --config config/TGN.yml --num_gpus 2 --rnd_edim 128 --rnd_ndim 128
EOF
