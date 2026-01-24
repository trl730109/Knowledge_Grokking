#!/bin/bash

# 知识预训练 + LoRA（基于 LLaMA-Factory）
# 支持参数传入的训练脚本

# 使用方法:
# bash train.sh <dataset_key> <dnn> <model_dir> <cuda_devices> [train_epochs] [output_dir]
# 例如: bash train.sh geo_history_1forward_1_1 qwen2.5-7b-instruct /workspace/tzc/Qwen/Qwen2.5-7B-Instruct 0,1,2,3

# 参数解析
dataset=${1:-${dataset}}
dnn=${2:-${dnn:-qwen2.5-7b-instruct}}
model_dir=${3:-${model_dir:-/workspace/tzc/Qwen/Qwen2.5-7B-Instruct}}
cuda_devices=${4:-${cuda_devices:-0,1,2,3}}
train_epochs=${5:-${train_epochs:-1.0}}
output_dir=${6:-${output_dir:-/workspace/tzc/Knowledge_Grokking/trained_models/${dnn}}}

# 检查必需参数
if [ -z "$dataset" ]; then
    echo "[ERROR] Dataset key is required!"
    echo "Usage: bash train.sh <dataset_key> [dnn] [model_dir] [cuda_devices] [train_epochs] [output_dir]"
    exit 1
fi

echo "========================================"
echo "Training Configuration"
echo "========================================"
echo "Dataset:      ${dataset}"
echo "Model:        ${dnn}"
echo "Model Dir:    ${model_dir}"
echo "CUDA Devices: ${cuda_devices}"
echo "Epochs:       ${train_epochs}"
echo "Output Dir:   ${output_dir}"
echo "========================================"

# 根据模型名称自动选择模板
template="llama3"
shopt -s nocasematch
if [[ "$dnn" == *qwen* ]] || [[ "$dnn" == *deepseek* ]]; then
    template="qwen"
elif [[ "$dnn" == *mistral* ]]; then
    template="mistral"
elif [[ "$dnn" == *llama2* ]]; then
    template="llama2"
elif [[ "$dnn" == *llama* ]]; then
    template="llama3"
fi
shopt -u nocasematch

echo "Using template: ${template}"

# 计算 GPU 数量
IFS=',' read -ra GPU_ARRAY <<< "$cuda_devices"
num_gpus=${#GPU_ARRAY[@]}
echo "Number of GPUs: ${num_gpus}"

# 获取脚本所在目录的上级目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# LLaMA-Factory 路径
LLAMAFACTORY_DIR="/workspace/tzc/LLaMA-Factory"

# 数据路径（指向 Knowledge_Grokking/processed_data）
DATA_DIR="${SCRIPT_DIR}/processed_data"

echo "[INFO] Starting training with LLaMA-Factory..."
echo "[INFO] Data directory: ${DATA_DIR}"

# 执行训练
cd "${LLAMAFACTORY_DIR}" || exit 1

CUDA_VISIBLE_DEVICES="${cuda_devices}" llamafactory-cli train \
    --stage pt \
    --do_train True \
    --model_name_or_path "${model_dir}" \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template "${template}" \
    --flash_attn auto \
    --dataset_dir "${DATA_DIR}" \
    --dataset "${dataset}" \
    --cutoff_len 4096 \
    --learning_rate 5e-05 \
    --num_train_epochs "${train_epochs}" \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --output_dir "${output_dir}" \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target all

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "[✔] Training completed successfully!"
    echo "[✔] Model saved to: ${output_dir}"
else
    echo "[✗] Training failed with exit code: ${exit_code}"
    exit $exit_code
fi
