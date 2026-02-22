############################################
# 1️⃣ 运行环境优化（高性能训练模式）
############################################
export CUDA_VISIBLE_DEVICES=2,3
export TORCH_DISTRIBUTED_DEBUG=OFF
export LOG_LEVEL=INFO
export NUMBA_DEBUG=0

# 🚀 关键加速项
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export NCCL_DEBUG=WARN 
export TORCH_CUDNN_BENCHMARK=1

# 显存管理
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:2


# NCCL 稳定 + 低日志
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0

# CPU 线程控制（根据机器核数改）
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

############################################
# 2️⃣ 启动分布式训练
############################################
/home/sqp17/miniconda3/envs/simple_py310/bin/python -O -m torch.distributed.run \
    --nproc_per_node=3 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_dist.py \
    --dataset LASTFM \
    --config /home/sqp17/Projects/original_tgl/config/dist/TGN.yml \
    --num_gpus 2 \
    --rnd_edim 128 \
    --rnd_ndim 128



############################################
# 1️⃣ 运行环境优化（高性能训练模式）
############################################
export CUDA_VISIBLE_DEVICES=2,3
export TORCH_DISTRIBUTED_DEBUG=OFF
export LOG_LEVEL=INFO
export NUMBA_DEBUG=0

# 🚀 关键加速项
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export NCCL_DEBUG=WARN 
export TORCH_CUDNN_BENCHMARK=1

# 显存管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL 稳定 + 低日志
export NCCL_DEBUG=VERSION
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0

# CPU 线程控制（根据机器核数改）
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
/home/sqp17/miniconda3/envs/simple_py310/bin/python \
    -m viztracer \
    --min_duration 200us \
    --output_file distributed_trace.json \
    --ignore_c_function \
    -m torch.distributed.run \
    --nproc_per_node=3 \
    --master_addr=127.0.0.1 \
    --master_port=29505 \
    train_dist.py \
    --data LASTFM \
    --config /home/sqp17/Projects/original_tgl/config/dist/TGN.yml  \