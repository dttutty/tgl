# TGL: A General Framework for Temporal Graph Training on Billion-Scale Graphs

This repo is based on https://github.com/amazon-science/tgl#.
Please refer to https://github.com/amazon-science/tgl# for installation and usage.
I use `benchmark_timing.sh` to measure the runtime of each stage, and `freshness.sh` to measure the importance of freshness.

## Environment

This fork now uses `uv` as the primary Python environment manager.

### Base environment for `simple_py310`

```bash
uv sync --python 3.10
```

That creates the project environment in `.venv`, installs the `sampler_core` extension through the project build backend, and matches the current `simple_py310` stack:

- `torch==2.2.2` from the CUDA 11.8 PyTorch index
- `dgl==2.4.0+cu118` from the official DGL wheel repo
- `torch-scatter==2.1.2+pt22cu118`
- `torchdata==0.7.1`

### Optional plotting dependencies

```bash
uv sync --python 3.10 --extra plot
```

### Runtime library setup

Some clusters ship an older system `libstdc++.so.6` or do not expose `libssl.so.3` by default. Before running the project, source [`scripts/uv-env.sh`](scripts/uv-env.sh) so the shell can prepend compatible runtime-library directories to `LD_LIBRARY_PATH` when needed.

```bash
source scripts/uv-env.sh
```

The script is a no-op when the current environment already provides suitable libraries. It currently patches lookup paths for:

- OpenSSL 3 (`libssl.so.3`, `libcrypto.so.3`)
- `libstdc++.so.6` with at least `GLIBCXX_3.4.26` for the DGL wheel used by this repo

If you prefer activating the virtual environment explicitly instead of `uv run`, use:

```bash
source .venv/bin/activate
source scripts/uv-env.sh
python train.py --data <NameOfYourDataset> --config <PathToConfigFile>
```

### Blackwell / NCCL experiment stack

The repository also contains a separate Blackwell-specific bootstrap flow in [`environment.blackwell-nccl.yml`](environment.blackwell-nccl.yml) and [`scripts/create_blackwell_nccl_env.sh`](scripts/create_blackwell_nccl_env.sh). That stack is not the default `uv` project environment above.

Do not use `uv sync` or the default `.venv` on Blackwell GPUs. That environment is pinned to the older `cu118` stack and will fail on `sm_120`.

Use the conda environment only as a source of the CUDA-enabled DGL package. The final runtime environment is `.venv-blackwell`, which the script assembles with `torch==2.10.0+cu128`, copies the DGL package and required native runtime libraries into place, and patches DGL so `import dgl` does not eagerly load the incompatible GraphBolt/distributed stack that TGL does not use.

Install it like this:

```bash
conda env create -f environment.blackwell-nccl.yml
SRC_DGL_PYTHON="$(conda run -n tgl-blackwell-nccl which python | tail -n 1)" \
  bash scripts/create_blackwell_nccl_env.sh
source .venv-blackwell/bin/activate
source scripts/uv-env.sh
```

Verify the final environment before training:

```bash
python - <<'PY'
import torch, dgl, torch_scatter, torchdata
print(torch.__version__, torch.version.cuda)
print(dgl.__version__)
print(torch_scatter.__version__)
print(torchdata.__version__)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

Expected result:

- `torch` reports `2.10.0+cu128`
- the GPU is detected as Blackwell with capability `(12, 0)`
- `import dgl` succeeds

If the bootstrap conda environment cannot `import dgl` after the pip-installed torch upgrade, that is expected. The runtime fix is applied to `.venv-blackwell` by [`scripts/patch_dgl_for_blackwell.py`](scripts/patch_dgl_for_blackwell.py).

For training on Blackwell, keep using `.venv-blackwell` directly:

```bash
source .venv-blackwell/bin/activate
source scripts/uv-env.sh
python train_non_timing_on_gpu.py --data <NameOfYourDataset> --config <PathToConfigFile>
```

### Running accuracy experiments on Blackwell

The launchers under `accuracy_experiment/` start nested Python commands. Some of
the shell wrappers, including
[`accuracy_experiment/compare_ap_tgl_vs_frost/0_run.sh`](accuracy_experiment/compare_ap_tgl_vs_frost/0_run.sh),
fall back to `.venv/bin/python` unless `PYTHON_BIN` is set explicitly.

For the Blackwell stack, do not launch these jobs with `uv run` or
`uv run --active`. Keep the runtime inside `.venv-blackwell` and point the
experiment wrapper at that interpreter explicitly:

```bash
source .venv-blackwell/bin/activate
source scripts/uv-env.sh
export PYTHON_BIN="$PWD/.venv-blackwell/bin/python"
```

The AP comparison experiment in
[`accuracy_experiment/compare_ap_tgl_vs_frost/0_run.yaml`](accuracy_experiment/compare_ap_tgl_vs_frost/0_run.yaml)
uses exactly 2 GPUs per job, so run it through
[`accuracy_experiment/run_on_gpu_pairs.py`](accuracy_experiment/run_on_gpu_pairs.py):

```bash
python accuracy_experiment/run_on_gpu_pairs.py \
  --script accuracy_experiment/compare_ap_tgl_vs_frost/0_run.sh
```

Optional flags:

- `--gpus 0,1` to restrict the scheduler to a specific adjacent GPU pair
- `-- --dataset CANPARL` to forward a dataset filter into the task script

Single-GPU experiment schedulers such as
[`accuracy_experiment/freshness/0_run.sh`](accuracy_experiment/freshness/0_run.sh)
should use
[`accuracy_experiment/run_on_one_gpu.py`](accuracy_experiment/run_on_one_gpu.py):

```bash
python accuracy_experiment/run_on_one_gpu.py \
  --script accuracy_experiment/freshness/0_run.sh
```

## Configuration Files

We provide example configuration files for five temporal GNN methods: JODIE, DySAT, TGAT, TGN and TGAT. The configuration files for single GPU training are located at `/config/` while the multiple GPUs training configuration files are located at `/config/dist/`.

The provided configuration files are all tested to be working. If you want to use your own network architecture, please refer to `/config/readme.yml` for the meaining of each entry in the yaml configuration file. As our framework is still under development, it possible that some combination of the confiruations will lead to bug. 

## Run

Currently, our framework only supports extrapolation setting (inference for the future).

### Single GPU Link Prediction
>source scripts/uv-env.sh && uv run python train.py --data \<NameOfYourDataset> --config \<PathToConfigFile>

### MultiGPU Link Prediction
>source scripts/uv-env.sh && uv run python -m torch.distributed.run --nproc_per_node=\<NumberOfGPUs+1> train_dist.py --data \<NameOfYourDataset> --config \<PathToConfigFile> --num_gpus \<NumberOfGPUs>

### Dynamic Node Classification

Currenlty, TGL only supports performing dynamic node classification using the dynamic node embedding generated in link prediction. 

For Single GPU models, directly run
>source scripts/uv-env.sh && uv run python train_node.py --data \<NameOfYourDATA> --config \<PathToConfigFile> --model \<PathToSavedModel>

For multi-GPU models, you need to first generate the dynamic node embedding
>source scripts/uv-env.sh && uv run python -m torch.distributed.run --nproc_per_node=\<NumberOfGPUs+1> extract_node_dist.py --data \<NameOfYourDataset> --config \<PathToConfigFile> --num_gpus \<NumberOfGPUs> --model \<PathToSavedModel>

After generating the node embeding for multi-GPU models, run
>source scripts/uv-env.sh && uv run python train_node.py --data \<NameOfYourDATA> --model \<PathToSavedModel>

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Cite

If you use TGL in a scientific publication, we would appreciate citations to the following paper:

```
@article{zhou2022tgl,
    title={{TGL}: A General Framework for Temporal GNN Training on Billion-Scale Graphs},
    author={Zhou, Hongkuan and Zheng, Da and Nisa, Israt and Ioannidis, Vasileios and Song, Xiang and Karypis, George},
    year = {2022},
    journal = {Proc. VLDB Endow.},
    volume = {15},
    number = {8},
}
```

## License

This project is licensed under the Apache-2.0 License.
