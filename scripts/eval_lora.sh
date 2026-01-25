#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

dnn=qwen2.5-7b-instruct
source "${SCRIPT_DIR}/env.sh"

lora_lists=(
    /workspace/tzc/Knowledge_Grokking/trained_models/qwen2.5-7b-instruct/0125_0336_ep10.0_lr1em4_r64/checkpoint-1410
    /workspace/tzc/Knowledge_Grokking/trained_models/qwen2.5-7b-instruct/0125_0426_ep15.0_lr1em4_r64/checkpoint-2115
    /workspace/tzc/Knowledge_Grokking/trained_models/qwen2.5-7b-instruct/0125_0540_ep20.0_lr5em5_r128/checkpoint-2820
    /workspace/tzc/Knowledge_Grokking/trained_models/qwen2.5-7b-instruct/0125_0735_ep10.0_lr2em4_r64/checkpoint-1410
    /workspace/tzc/Knowledge_Grokking/trained_models/qwen2.5-7b-instruct/0125_0824_ep10.0_lr1em4_r128/checkpoint-1410
)
test_datasets=geo
cuda_devices=0,1

model_name="${dnn}"
model_path="${model_dir}"

IFS=',' read -ra GPU_ARRAY <<< "$cuda_devices"
tensor_parallel_size=${#GPU_ARRAY[@]}

if [ "$test_datasets" == "all" ]; then
    datasets=(bio brand creative game geo history mat)
else
    IFS=',' read -ra datasets <<< "$test_datasets"
fi

# 遍历所有 LoRA 路径
for lora_path in "${lora_lists[@]}"; do
    # 从路径中提取标识符（倒数第二层目录名）
    # 例如: /path/to/0125_0336_ep10.0_lr1em4_r64/checkpoint-1410 -> 0125_0336_ep10.0_lr1em4_r64
    lora_identifier=$(basename "$(dirname "${lora_path}")")
    output_dir_prefix="${PROJECT_DIR}/output/lora/${model_name}/${lora_identifier}"
    
    # 生成统一的 date_str 用于本次评估
    eval_date_str=$(date +%m%d_%H%M)
    
    for dataset in "${datasets[@]}"; do
        dataset_output_dir="${output_dir_prefix}/${dataset}"
        CUDA_VISIBLE_DEVICES=${cuda_devices} python3 ${PROJECT_DIR}/eval_grokking_vLLM.py \
            --model_name ${model_name} \
            --model_path ${model_path} \
            --lora_path ${lora_path} \
            --test_datasets ${dataset} \
            --base_data_dir ${PROJECT_DIR}/test_data \
            --tensor_parallel_size ${tensor_parallel_size} \
            --output_dir ${dataset_output_dir} \
            --date_str ${eval_date_str}
    done
    
    # 汇总所有数据集的结果
    overall_file="${output_dir_prefix}/overall.txt"
    mkdir -p "${output_dir_prefix}"
    
    {
        echo "Overall Evaluation Summary"
        echo "Model: ${model_name}"
        echo "LoRA Path: ${lora_path}"
        echo "Date: ${eval_date_str}"
        echo "Test Datasets: ${test_datasets}"
        echo "========================================"
        echo ""
        
        total_avg=0
        domain_count=0
        
        for dataset in "${datasets[@]}"; do
            results_file="${output_dir_prefix}/${dataset}/${eval_date_str}/${dataset}/results.txt"
            if [ -f "${results_file}" ]; then
                echo "[${dataset^^}]"
                echo "----------------------------------------"
                grep -E "^(direct_qa|inverse_qa|mcq|multihop|scenario|tool|spatial)" "${results_file}" || true
                avg_score=$(grep "^Average:" "${results_file}" | awk '{print $2}')
                if [ -n "$avg_score" ]; then
                    echo "Average: ${avg_score}"
                    total_avg=$(awk "BEGIN {printf \"%.4f\", ${total_avg} + ${avg_score}}")
                    domain_count=$((domain_count + 1))
                fi
                echo ""
            fi
        done
        
        echo "========================================"
        if [ $domain_count -gt 0 ]; then
            overall_avg=$(awk "BEGIN {printf \"%.4f\", ${total_avg} / ${domain_count}}")
            echo "Overall Average (across ${domain_count} domains): ${overall_avg}"
        fi
        echo "========================================"
    } > "${overall_file}"
done

