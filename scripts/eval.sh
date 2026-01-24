#!/bin/bash

# 评估脚本 - 支持参数传入
# 使用方法:
# bash eval.sh <model_name> <model_path> <cuda_devices> [lora_path] [test_datasets] [tensor_parallel_size]
# 例如: bash eval.sh qwen2.5-7b-instruct /workspace/tzc/Qwen/Qwen2.5-7B-Instruct 0,1,2,3 /path/to/lora all 4

# 参数解析
model_name=${1:-${model_name}}
model_path=${2:-${model_path}}
cuda_devices=${3:-${cuda_devices:-0,1,2,3}}
lora_path=${4:-${lora_path}}  # 可选，如果为空则评估 base model
test_datasets=${5:-${test_datasets:-all}}
tensor_parallel_size=${6:-${tensor_parallel_size}}

# 检查必需参数
if [ -z "$model_name" ] || [ -z "$model_path" ]; then
    echo "[ERROR] Model name and model path are required!"
    echo "Usage: bash eval.sh <model_name> <model_path> [cuda_devices] [lora_path] [test_datasets] [tensor_parallel_size]"
    exit 1
fi

# 如果没有指定 tensor_parallel_size，从 cuda_devices 计算
if [ -z "$tensor_parallel_size" ]; then
    IFS=',' read -ra GPU_ARRAY <<< "$cuda_devices"
    tensor_parallel_size=${#GPU_ARRAY[@]}
fi

echo "========================================"
echo "Evaluation Configuration"
echo "========================================"
echo "Model Name:     ${model_name}"
echo "Model Path:     ${model_path}"
echo "LoRA Path:      ${lora_path:-None (Base Model)}"
echo "Test Datasets:  ${test_datasets}"
echo "CUDA Devices:   ${cuda_devices}"
echo "Tensor Parallel: ${tensor_parallel_size}"
echo "========================================"

# 获取脚本所在目录的上级目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 构建命令
CMD="CUDA_VISIBLE_DEVICES=${cuda_devices} python3 ${SCRIPT_DIR}/eval_grokking.py \
    --model_name ${model_name} \
    --model_path ${model_path} \
    --test_datasets ${test_datasets} \
    --base_data_dir ${SCRIPT_DIR}/test_data \
    --tensor_parallel_size ${tensor_parallel_size}"

# 如果提供了 lora_path，添加到命令中
if [ -n "$lora_path" ]; then
    CMD="${CMD} --lora_path ${lora_path}"
fi

echo "[INFO] Starting evaluation..."
echo "[CMD] ${CMD}"
echo ""

# 执行评估
eval ${CMD}

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "[✔] Evaluation completed successfully!"
else
    echo "[✗] Evaluation failed with exit code: ${exit_code}"
    exit $exit_code
fi
