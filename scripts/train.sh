#!/bin/bash
dataset=${1:-${dataset}}
dnn=${2:-${dnn:-qwen2.5-7b-instruct}}
model_dir=${3:-${model_dir:-/workspace/tzc/Qwen/Qwen2.5-7B-Instruct}}
cuda_devices=${4:-${cuda_devices:-0,1,2,3}}
train_epochs=${5:-${train_epochs:-1.0}}
output_dir=${6:-${output_dir:-/workspace/tzc/Knowledge_Grokking/trained_models/${dnn}}}
learning_rate=${7:-${learning_rate:-1e-4}}
lora_rank=${8:-${lora_rank:-64}}

# 检查必需参数
if [ -z "$dataset" ]; then
    echo "[ERROR] Dataset key is required!"
    echo "Usage: bash train.sh <dataset_key> [dnn] [model_dir] [cuda_devices] [train_epochs] [output_dir] [learning_rate] [lora_rank]"
    exit 1
fi

# 计算 lora_alpha = lora_rank * 2
lora_alpha=$((lora_rank * 2))

echo "========================================"
echo "Training Configuration"
echo "========================================"
echo "Dataset:      ${dataset}"
echo "Model:        ${dnn}"
echo "Model Dir:    ${model_dir}"
echo "CUDA Devices: ${cuda_devices}"
echo "Epochs:       ${train_epochs}"
echo "Learning Rate: ${learning_rate}"
echo "LoRA Rank:    ${lora_rank}"
echo "LoRA Alpha:   ${lora_alpha} (auto: rank * 2)"
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

# # LLaMA-Factory 路径
# LLAMAFACTORY_DIR="/workspace/tzc/LLaMA-Factory"

# 数据路径（指向 Knowledge_Grokking/tmp - 临时合成数据集）
DATA_DIR="${SCRIPT_DIR}/tmp"

echo "[INFO] Starting training with LLaMA-Factory..."
echo "[INFO] Data directory: ${DATA_DIR}"

# 计算 warmup_steps
# 动态计算 steps_per_epoch = 数据集数量 / (batch_size * gradient_accumulation_steps * num_gpus)
per_device_batch_size=2
gradient_accumulation_steps=8

# 使用 Python 脚本计算 steps_per_epoch
dataset_info_file="${DATA_DIR}/dataset_info.json"
steps_per_epoch=$(python3 "${SCRIPT_DIR}/scripts/calculate_steps_per_epoch.py" \
    "${dataset}" \
    "${DATA_DIR}" \
    "${dataset_info_file}" \
    "${per_device_batch_size}" \
    "${gradient_accumulation_steps}" \
    "${num_gpus}")

# 如果脚本返回失败，使用默认值
if [ -z "${steps_per_epoch}" ] || [ "${steps_per_epoch}" -lt 1 ]; then
    echo "[WARNING] Failed to calculate steps_per_epoch, using default value 140"
    steps_per_epoch=140
fi

total_steps=$(awk "BEGIN {printf \"%.0f\", ${train_epochs} * ${steps_per_epoch}}")
# 计算 warmup_steps = total_steps * 10%
warmup_steps=$(awk "BEGIN {printf \"%.0f\", ${total_steps} * 0.1}" 2>/dev/null)
# 如果计算失败，设置为 0
if [ -z "${warmup_steps}" ] || [ "${warmup_steps}" -lt 0 ]; then
    warmup_steps=0
fi
echo "[INFO] Calculated warmup_steps: ${warmup_steps} (total_steps=${total_steps}, steps_per_epoch=${steps_per_epoch}, epochs=${train_epochs})"


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
    --learning_rate "${learning_rate}" \
    --num_train_epochs "${train_epochs}" \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_only_model true \
    --save_strategy epoch \
    --save_total_limit 1 \
    --warmup_steps "${warmup_steps}" \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --output_dir "${output_dir}" \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --lora_rank "${lora_rank}" \
    --lora_alpha "${lora_alpha}" \
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
