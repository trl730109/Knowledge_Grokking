#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# dnn=qwen2.5-7b-instruct
dnn=qwen3-8b
source "${SCRIPT_DIR}/env.sh"

test_datasets=${1:-all}  # 默认测全部，可传入 geo 或 geo,game,history
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

# 设置输出目录前缀，Python 脚本会在这个前缀下为每个 domain 创建子目录
output_dir_prefix="${PROJECT_DIR}/output/base_model/${model_name}"

# 生成统一的 date_str 用于本次评估
eval_date_str=$(date +%m%d_%H%M)

for dataset in "${datasets[@]}"; do
    echo "========== Evaluating: ${dataset} =========="
    dataset_output_dir="${output_dir_prefix}/${dataset}"
    CUDA_VISIBLE_DEVICES=${cuda_devices} python3 ${PROJECT_DIR}/eval_grokking_vLLM.py \
        --model_name ${model_name} \
        --model_path ${model_path} \
        --test_datasets ${dataset} \
        --base_data_dir ${PROJECT_DIR}/test_data \
        --tensor_parallel_size ${tensor_parallel_size} \
        --output_dir ${dataset_output_dir} \
        --date_str ${eval_date_str}
    
    if [ $? -ne 0 ]; then
        echo "[✗] Failed: ${dataset}"
        exit 1
    fi
done

# 汇总所有数据集的结果
echo ""
echo "========== Generating Overall Summary =========="
overall_file="${output_dir_prefix}/overall.txt"
mkdir -p "${output_dir_prefix}"

{
    echo "Overall Evaluation Summary"
    echo "Model: ${model_name}"
    echo "Date: ${eval_date_str}"
    echo "Test Datasets: ${test_datasets}"
    echo "========================================"
    echo ""
    
    total_avg=0
    domain_count=0
    
    for dataset in "${datasets[@]}"; do
        results_file="${output_dir_prefix}/${dataset}/${eval_date_str}/${dataset}/results.txt"
        if [ -f "${results_file}" ]; then
            echo "[${dataset.upper()}]"
            echo "----------------------------------------"
            # 提取并显示各任务分数
            grep -E "^(direct_qa|inverse_qa|mcq|multihop|scenario|tool|spatial)" "${results_file}" || true
            # 提取平均分
            avg_score=$(grep "^Average:" "${results_file}" | awk '{print $2}')
            if [ -n "$avg_score" ]; then
                echo "Average: ${avg_score}"
                total_avg=$(awk "BEGIN {printf \"%.4f\", ${total_avg} + ${avg_score}}")
                domain_count=$((domain_count + 1))
            fi
            echo ""
        else
            echo "[${dataset.upper()}] - Results file not found: ${results_file}"
            echo ""
        fi
    done
    
    echo "========================================"
    if [ $domain_count -gt 0 ]; then
        overall_avg=$(awk "BEGIN {printf \"%.4f\", ${total_avg} / ${domain_count}}")
        echo "Overall Average (across ${domain_count} domains): ${overall_avg}"
    else
        echo "No valid results found"
    fi
    echo "========================================"
} > "${overall_file}"

echo "[✔] Overall summary saved to: ${overall_file}"
cat "${overall_file}"

echo ""
echo "[✔] All evaluations completed!"
