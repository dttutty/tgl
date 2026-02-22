#!/bin/bash

# ==========================================
# 1. 基础环境设置 (保持您提供的优化参数)
# ==========================================
export CUDA_VISIBLE_DEVICES=0,1  # 注意：您下面命令用了3个进程，所以这里最好放开足够的卡
export TORCH_DISTRIBUTED_DEBUG=OFF
export LOG_LEVEL=INFO
export NUMBA_DEBUG=0

# 🚀 关键加速项
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export NCCL_DEBUG=WARN 
export TORCH_CUDNN_BENCHMARK=1

# 显存管理
export PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:2

# NCCL 稳定 + 低日志
export NCCL_DEBUG=VERSION
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0

# CPU 线程控制
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Python 解释器路径 (根据您的路径修改)
PYTHON_BIN="/home/sqp17/miniconda3/envs/simple_py310/bin/python"

# ==========================================
# 2. 实验循环配置
# ==========================================
TOTAL_RUNS=20
LOG_DIR="logs_experiment_$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="${LOG_DIR}/summary_report.txt"

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "============================================"
echo "🚀 开始执行 $TOTAL_RUNS 次重复实验"
echo "📂 日志目录: $LOG_DIR"
echo "============================================"

# 初始化汇总文件
echo "Run_ID,Test_AP,Test_AUC" > "$SUMMARY_FILE"

for ((i=1; i<=TOTAL_RUNS; i++))
do
    # 格式化运行编号 (例如 01, 02)
    RUN_ID=$(printf "%02d" $i)
    LOG_FILE="${LOG_DIR}/run_${RUN_ID}.log"
    
    echo "--------------------------------------------"
    echo "[$(date +%T)] 正在执行第 $i / $TOTAL_RUNS 次实验..."
    
    # --- 运行命令 ---
    # 注意：根据您的输入，nproc_per_node=3 但 num_gpus=2。
    # 通常 num_gpus 应该匹配 worker 数量，或者您代码里有特殊逻辑。
    # 这里完全照搬您的命令结构。
    $PYTHON_BIN -m torch.distributed.run \
        --nproc_per_node=3 \
        --master_addr=127.0.0.1 \
        --master_port=$((29600 + i)) \
        train_dist.py \
        --data LASTFM \
        --config /home/sqp17/Projects/original_tgl/config/dist/TGN_128_600.yml \
        --num_gpus 2 \
        --rnd_edim 128 \
        --rnd_ndim 128 \
        > "$LOG_FILE" 2>&1
    
    # 检查上一条命令是否成功
    if [ $? -ne 0 ]; then
        echo "❌ 第 $i 次实验失败! 请查看 $LOG_FILE"
        # 可以选择 continue 或 exit
    else
        echo "✅ 第 $i 次实验完成。"
        
        # --- 自动抓取结果 (假设日志中有特定格式) ---
        # 假设您的代码最后打印: test ap:0.9xxx  test auc:0.9xxx
        # 下面的 grep 命令尝试提取最后出现的数值
        
        # 提取 AP (寻找 'test ap:' 后的数字)
        AP=$(grep -oP "test ap:\K[0-9.]+" "$LOG_FILE" | tail -1)
        # 提取 AUC (寻找 'test auc:' 后的数字)
        AUC=$(grep -oP "test auc:\K[0-9.]+" "$LOG_FILE" | tail -1)
        
        # 如果您按照我们之前的修改增加了 Global/Local 对比打印，
        # 请修改这里的 grep 逻辑，例如匹配 "Global\(Correct\) AP:"
        # 示例：
        # AP=$(grep -oP "Global\(Correct\) AP: \K[0-9.]+" "$LOG_FILE" | tail -1)
        
        echo "   提取结果 -> AP: ${AP:-N/A}, AUC: ${AUC:-N/A}"
        echo "$RUN_ID,$AP,$AUC" >> "$SUMMARY_FILE"
    fi
    
    # 稍微休息一下，防止 GPU 过热或端口未释放
    sleep 5
done

# ==========================================
# 3. 计算平均值
# ==========================================
echo "--------------------------------------------"
echo "📊 正在计算平均值..."

# 使用 awk 计算 CSV 中第2列(AP)和第3列(AUC)的平均值
AVG_AP=$(awk -F',' 'NR>1 && $2!="" {sum+=$2; count++} END {if (count>0) print sum/count; else print "0"}' "$SUMMARY_FILE")
AVG_AUC=$(awk -F',' 'NR>1 && $3!="" {sum+=$3; count++} END {if (count>0) print sum/count; else print "0"}' "$SUMMARY_FILE")

echo "============================================"
echo "🏁 所有实验结束！"
echo "平均 Test AP : $AVG_AP"
echo "平均 Test AUC: $AVG_AUC"
echo "完整报告路径 : $SUMMARY_FILE"
echo "============================================"