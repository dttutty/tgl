# TGL: A General Framework for Temporal Graph Training on Billion-Scale Graphs

This repo is based on https://github.com/amazon-science/tgl#.
Please refer to https://github.com/amazon-science/tgl# for installation and usage.
I use `benchmark_timing.sh` to measure the runtime of each stage, and `freshness.sh` to measure the importance of freshness.

## Configuration Files

We provide example configuration files for five temporal GNN methods: JODIE, DySAT, TGAT, TGN and TGAT. The configuration files for single GPU training are located at `/config/` while the multiple GPUs training configuration files are located at `/config/dist/`.

The provided configuration files are all tested to be working. If you want to use your own network architecture, please refer to `/config/readme.yml` for the meaining of each entry in the yaml configuration file. As our framework is still under development, it possible that some combination of the confiruations will lead to bug. 

## Run

Currently, our framework only supports extrapolation setting (inference for the future).

### Single GPU Link Prediction
>python train.py --data \<NameOfYourDataset> --config \<PathToConfigFile>

### MultiGPU Link Prediction
>python -m torch.distributed.launch --nproc_per_node=\<NumberOfGPUs+1> train_dist.py --data \<NameOfYourDataset> --config \<PathToConfigFile> --num_gpus \<NumberOfGPUs>

### Dynamic Node Classification

Currenlty, TGL only supports performing dynamic node classification using the dynamic node embedding generated in link prediction. 

For Single GPU models, directly run
>python train_node.py --data \<NameOfYourDATA> --config \<PathToConfigFile> --model \<PathToSavedModel>

For multi-GPU models, you need to first generate the dynamic node embedding
>python -m torch.distributed.launch --nproc_per_node=\<NumberOfGPUs+1> extract_node_dist.py --data \<NameOfYourDataset> --config \<PathToConfigFile> --num_gpus \<NumberOfGPUs> --model \<PathToSavedModel>

After generating the node embeding for multi-GPU models, run
>python train_node.py --data \<NameOfYourDATA> --model \<PathToSavedModel>

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
