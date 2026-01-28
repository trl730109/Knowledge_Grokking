#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

dnn=qwen3-8b
source "${SCRIPT_DIR}/env.sh"

lora_configs=(
"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/onestep_0126_1641_ep10.0_lr2em4_r64:bio:onestep"
"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/onestep_0126_1704_ep10.0_lr2em4_r64:brand:onestep"
"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/onestep_0126_1724_ep10.0_lr2em4_r64:creative:onestep"
"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/onestep_0126_1739_ep10.0_lr2em4_r64:game:onestep"
"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/onestep_0126_1800_ep10.0_lr2em4_r64:geo:onestep"
"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/onestep_0126_1823_ep10.0_lr2em4_r64:history:onestep"
"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/onestep_0126_1842_ep10.0_lr2em4_r64:mat:onestep"

"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/twostep_0126_1904_ep10.0_lr2em4_r64:bio:twostep"
"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/twostep_0126_1942_ep10.0_lr2em4_r64:brand:twostep"
"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/twostep_0126_2010_ep10.0_lr2em4_r64:creative:twostep"
"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/twostep_0126_2040_ep10.0_lr2em4_r64:game:twostep"
"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/twostep_0126_2118_ep10.0_lr2em4_r64:geo:twostep"
"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/twostep_0126_2157_ep10.0_lr2em4_r64:history:twostep"
"/workspace/tzc/Knowledge_Grokking/trained_models/qwen3-8b/ablation_type/twostep_0126_2233_ep10.0_lr2em4_r64:mat:twostep"
)

types=ablation_type
date_str=$(date +%m%d_%H%M)
cuda_devices=0,1

model_name="${dnn}"
model_path="${model_dir}"

IFS=',' read -ra GPU_ARRAY <<< "$cuda_devices"
tensor_parallel_size=${#GPU_ARRAY[@]}

# 遍历所有 LoRA 配置
for config in "${lora_configs[@]}"; do
    # 解析配置：格式为 "lorapath:test_datasets:step_type"
    IFS=':' read -r lora_path test_datasets step_type <<< "$config"
    
    if [ -z "$lora_path" ] || [ -z "$test_datasets" ] || [ -z "$step_type" ]; then
        echo "[ERROR] Invalid config format: ${config}"
        echo "Expected format: lorapath:test_datasets:step_type"
        continue
    fi
    
    # 从 step_type 中提取 one 或 two
    # onestep -> one, twostep -> two
    if [[ "$step_type" == "onestep" ]]; then
        step_prefix="one"
    elif [[ "$step_type" == "twostep" ]]; then
        step_prefix="two"
    else
        echo "[ERROR] Invalid step_type: ${step_type}. Expected 'onestep' or 'twostep'"
        continue
    fi
    
    # 解析测试数据集
    if [ "$test_datasets" == "all" ]; then
        datasets=(bio brand creative game geo history mat)
    else
        IFS=',' read -ra datasets <<< "$test_datasets"
    fi
    # 从路径中提取标识符（倒数第二层目录名）
    # 例如: /path/to/0125_0336_ep10.0_lr1em4_r64/checkpoint-1410 -> 0125_0336_ep10.0_lr1em4_r64
    lora_identifier=$(basename "$(dirname "${lora_path}")")
    output_dir_prefix="${PROJECT_DIR}/output/${types}/${model_name}/${step_prefix}/${date_str}/${lora_identifier}"
    
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

