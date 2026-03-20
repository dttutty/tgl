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

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${REPO_DIR}/scripts/uv-env.sh"
ENV_DIR="${ENV_DIR:-${REPO_DIR}/.venv-blackwell}"
PYTHON_REQ="${PYTHON_REQ:-3.10}"
SRC_DGL_PYTHON="${SRC_DGL_PYTHON:-}"

resolve_source_dgl_python() {
  if [[ -n "${SRC_DGL_PYTHON}" ]]; then
    printf '%s\n' "${SRC_DGL_PYTHON}"
    return
  fi

  if ! command -v python >/dev/null 2>&1; then
    echo "Could not find a source Python for DGL bootstrap." >&2
    echo "Set SRC_DGL_PYTHON to a Python 3.10 executable that already imports CUDA-enabled dgl." >&2
    exit 1
  fi

  printf '%s\n' "$(command -v python)"
}

SRC_DGL_PYTHON="$(resolve_source_dgl_python)"

if [[ ! -x "${SRC_DGL_PYTHON}" ]]; then
  echo "Source DGL Python not found: ${SRC_DGL_PYTHON}" >&2
  exit 1
fi

echo "Syncing uv environment"
uv venv --python "${PYTHON_REQ}" "${ENV_DIR}"

PYTHON_BIN="${ENV_DIR}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "uv did not create the destination environment: ${ENV_DIR}" >&2
  exit 1
fi

if [[ "$(readlink -f "${SRC_DGL_PYTHON}")" == "$(readlink -f "${PYTHON_BIN}")" ]]; then
  echo "Source and destination Python are the same: ${PYTHON_BIN}" >&2
  echo "Activate the old working env first, or set SRC_DGL_PYTHON explicitly." >&2
  exit 1
fi

echo "Installing the Blackwell/NCCL Python stack"
uv pip install --python "${PYTHON_BIN}" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0+cu128

uv pip install --python "${PYTHON_BIN}" \
  --extra-index-url https://pypi.org/simple \
  --find-links https://data.pyg.org/whl/torch-2.10.0+cu128.html \
  numpy \
  packaging \
  pandas \
  psutil \
  pybind11 \
  pydantic \
  pyyaml \
  requests \
  scikit-learn \
  scipy \
  setuptools \
  torch-scatter==2.1.2+pt210cu128 \
  torchdata==0.9.0 \
  tqdm

echo "Checking source DGL environment"
"${SRC_DGL_PYTHON}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        "This script expects the source DGL environment to use Python 3.10."
    )

try:
    import dgl  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "Source Python cannot import dgl. Set SRC_DGL_PYTHON to a working env."
    ) from exc
PY

SRC_SITE_PACKAGES="$("${SRC_DGL_PYTHON}" - <<'PY'
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
SRC_DGL_METADATA_DIR="$(
  find "${SRC_SITE_PACKAGES}" -maxdepth 1 -type d \( -name 'dgl-*.dist-info' -o -name 'dgl-*.egg-info' \) | head -n 1
)"

if [[ ! -d "${SRC_DGL_DIR}" ]]; then
  echo "Could not find ${SRC_DGL_DIR}" >&2
  exit 1
fi

if [[ -z "${SRC_DGL_METADATA_DIR}" ]]; then
  echo "Could not find dgl metadata under ${SRC_SITE_PACKAGES}" >&2
  exit 1
fi

echo "Copying CUDA-enabled DGL from ${SRC_DGL_PYTHON}"
rm -rf "${DST_SITE_PACKAGES}/dgl"
find "${DST_SITE_PACKAGES}" -maxdepth 1 -type d -name 'dgl-*.dist-info' -exec rm -rf {} +
find "${DST_SITE_PACKAGES}" -maxdepth 1 -type d -name 'dgl-*.egg-info' -exec rm -rf {} +
cp -a "${SRC_DGL_DIR}" "${DST_SITE_PACKAGES}/"
cp -a "${SRC_DGL_METADATA_DIR}" "${DST_SITE_PACKAGES}/"

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

Distributed run:
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
