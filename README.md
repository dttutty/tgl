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

If you run with deterministic PyTorch algorithms enabled, set the CuBLAS workspace mode before starting Python:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

This avoids the CUDA-side deterministic-algorithm error raised by `torch.use_deterministic_algorithms(True)` when `nn.Linear` and other CuBLAS-backed ops execute on GPU.

If you prefer activating the virtual environment explicitly instead of `uv run`, use:

```bash
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/uv-env.sh
python train.py --data <NameOfYourDataset> --config <PathToConfigFile>
```

### Blackwell / NCCL experiment stack

The repository also contains a separate Blackwell-specific experiment stack in [`environment.blackwell-nccl.yml`](environment.blackwell-nccl.yml) and [`scripts/create_blackwell_nccl_env.sh`](scripts/create_blackwell_nccl_env.sh). That stack is not the default `uv` project environment above.

If you need that exact experimental stack, bootstrap it separately:

```bash
conda activate simple_py310
bash scripts/create_blackwell_nccl_env.sh
```

If you do not want to activate the old conda env first, set `SRC_DGL_PYTHON=/path/to/simple_py310/bin/python` before running the script.

## Configuration Files

We provide example configuration files for five temporal GNN methods: JODIE, DySAT, TGAT, TGN and TGAT. The configuration files for single GPU training are located at `/config/` while the multiple GPUs training configuration files are located at `/config/dist/`.

The provided configuration files are all tested to be working. If you want to use your own network architecture, please refer to `/config/readme.yml` for the meaining of each entry in the yaml configuration file. As our framework is still under development, it possible that some combination of the confiruations will lead to bug. 

## Run

Currently, our framework only supports extrapolation setting (inference for the future).
The commands below assume you have already run `export CUBLAS_WORKSPACE_CONFIG=:4096:8` in the current shell if you keep the default deterministic-training behavior in the provided scripts.

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
