#!/bin/bash

# 知识预训练 + LoRA（基于 LLaMA-Factory）

# set -x
dnn=${dnn:-qwen2.5-7b-instruct}
model_dir=${model_dir:-/workspace/tzc/Qwen/Qwen2.5-7B-Instruct}

# 使用 Knowledge_Grokking 中定义的预训练数据集（见 train_data/dataset_info.json）
# 可以通过环境变量 dataset 覆盖，例如：dataset=geo_1_attribute
dataset=${dataset:-geo_1_attribute}

# 输出目录放在当前项目下
output_dir=${output_dir:-/workspace/tzc/Knowledge_Grokking/trained_models/${dnn}}

train_epochs=${train_epochs:-1.0}
cuda_devices=${cuda_devices:-0,1,2,3}

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

echo "Pretrain dataset: ${dataset} model_dir: ${model_dir} output_dir: ${output_dir} template: ${template}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

CUDA_VISIBLE_DEVICES=${cuda_devices} llamafactory-cli train \
    --model_name_or_path ${model_dir} \
    --trust_remote_code \
    --stage pt \
    --do_train \
    --packing false \
    --finetuning_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --lora_target all \
    --dataset ${dataset} \
    --dataset_dir ./train_data \
    --template ${template} \
    --cutoff_len 2048 \
    --max_samples ${max_samples:-1000} \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir ${output_dir} \
    --logging_steps 1 \
    --overwrite_output_dir \
    --save_only_model false \
    --save_strategy no \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --num_train_epochs ${train_epochs} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000