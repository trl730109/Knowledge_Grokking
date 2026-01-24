#!/bin/bash
set -euo pipefail

# 默认配置
DEFAULT_DNN="qwen2.5-7b-instruct"
DEFAULT_CUDA_DEVICES="0,1,2,3"

# 实验配置列表
# 格式: categories:rewrite_types:ratios:dnn:cuda_devices
# 注意：dataset_key 会根据 categories, rewrite_types, ratios 自动生成
experiments=(
  "geo:1_forward,2_premise:1:1:qwen2.5-7b-instruct:${DEFAULT_CUDA_DEVICES}"
  # "geo,game:1_forward,1_inverse:1:1:qwen2.5-7b-instruct:${DEFAULT_CUDA_DEVICES}"
  # 添加更多实验配置...
  # "bio,brand:2_premise,3_conclusion:2:1:qwen2.5-14b-instruct:${DEFAULT_CUDA_DEVICES}"
)

script_date=$(date +%Y%m%d_%H%M%S)

run_experiment() {
  local categories="$1"
  local rewrite_types="$2"
  local ratios="$3"
  local dataset_name="$4"  # 这个参数现在不使用，会自动生成
  local dnn="$5"
  local cuda_devices="$6"
  
  echo "========================================"
  echo "[RUN] Experiment Configuration:"
  echo "  Categories:    ${categories}"
  echo "  Rewrite Types: ${rewrite_types}"
  echo "  Ratios:        ${ratios}"
  echo "  Model:         ${dnn}"
  echo "  CUDA Devices:  ${cuda_devices}"
  echo "========================================"

  # 导出环境变量
  export dnn
  export cuda_devices
  
  # 加载环境配置，获取 model_dir
  source "$(dirname "$0")/env.sh"
  
  echo "[INFO] Model directory: ${model_dir}"

  # 获取项目根目录
  local SCRIPT_DIR
  SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
  
  # Step 1: 数据合成
  echo ""
  echo "[STEP 1/3] Synthesizing dataset..."
  local syn_output
  syn_output=$(python "${SCRIPT_DIR}/syn_and_train.py" \
    --categories "${categories}" \
    --rewrite_types "${rewrite_types}" \
    --ratios "${ratios}" 2>&1)
  
  if [ $? -ne 0 ]; then
    echo "[ERROR] ✗ Data synthesis failed"
    echo "${syn_output}"
    return 1
  fi
  
  echo "${syn_output}"
  
  # 从输出中提取 dataset_key
  local dataset_key
  dataset_key=$(echo "${syn_output}" | grep "^DATASET_KEY=" | cut -d'=' -f2)
  
  if [ -z "${dataset_key}" ]; then
    echo "[ERROR] Failed to extract dataset key from synthesis output"
    return 1
  fi
  
  echo "[INFO] ✓ Data synthesis completed"
  echo "[INFO] Dataset Key: ${dataset_key}"
  
  # Step 2: 训练
  echo ""
  echo "[STEP 2/3] Training model..."
  local train_epochs="${train_epochs:-1.0}"
  local output_dir="${SCRIPT_DIR}/trained_models/${dnn}"
  
  if bash "${SCRIPT_DIR}/scripts/train.sh" \
    "${dataset_key}" \
    "${dnn}" \
    "${model_dir}" \
    "${cuda_devices}" \
    "${train_epochs}" \
    "${output_dir}"; then
    echo "[INFO] ✓ Training completed successfully"
  else
    echo "[ERROR] ✗ Training failed"
    return 1
  fi
  
  # 训练完成后的 LoRA 路径
  local lora_path="${output_dir}"
  
  # Step 3: 评估
  echo ""
  echo "[STEP 3/3] Evaluating model..."
  echo "[INFO] Evaluating on trained categories: ${categories}"
  
  # 计算 tensor_parallel_size
  local tensor_parallel_size
  tensor_parallel_size=$(echo "${cuda_devices}" | tr ',' '\n' | wc -l)
  
  if bash "${SCRIPT_DIR}/scripts/eval.sh" \
    "${dnn}" \
    "${model_dir}" \
    "${cuda_devices}" \
    "${lora_path}" \
    "${categories}" \
    "${tensor_parallel_size}"; then
    echo "[INFO] ✓ Evaluation completed successfully"
  else
    echo "[WARNING] ✗ Evaluation failed, but continuing..."
  fi
  
  echo ""
  echo "[DONE] Experiment completed: ${dataset_key}"
  echo "========================================"
  echo ""
}

# 主程序
total_experiments=${#experiments[@]}
echo "=================================================================="
echo "Knowledge Grokking Experiment Runner"
echo "Script Date: ${script_date}"
echo "Total experiments: ${total_experiments}"
echo "=================================================================="
echo ""

for exp_idx in "${!experiments[@]}"; do
  exp_config="${experiments[$exp_idx]}"
  exp_num=$((exp_idx + 1))

  # 解析配置：categories:rewrite_types:ratios:dnn:cuda_devices
  IFS=':' read -r categories_cfg rewrite_types_cfg ratios_cfg dnn_cfg cuda_devices_cfg <<< "$exp_config"

  # 应用默认值
  if [ -z "$dnn_cfg" ] || [ "$dnn_cfg" == "" ]; then
    dnn_cfg="$DEFAULT_DNN"
  fi
  if [ -z "$cuda_devices_cfg" ] || [ "$cuda_devices_cfg" == "" ]; then
    cuda_devices_cfg="$DEFAULT_CUDA_DEVICES"
  fi

  echo "=================================================================="
  echo "Experiment ${exp_num}/${total_experiments}"
  echo "  Categories:    ${categories_cfg}"
  echo "  Rewrite Types: ${rewrite_types_cfg}"
  echo "  Ratios:        ${ratios_cfg}"
  echo "  Model:         ${dnn_cfg}"
  echo "  CUDA Devices:  ${cuda_devices_cfg}"
  echo "=================================================================="

  # dataset_name 参数设为空，会在 run_experiment 中自动生成
  if run_experiment "${categories_cfg}" "${rewrite_types_cfg}" "${ratios_cfg}" "" "${dnn_cfg}" "${cuda_devices_cfg}"; then
    echo "✓ Experiment ${exp_num} completed successfully"
  else
    echo "✗ Experiment ${exp_num} failed with exit code $?"
    echo "Continuing with next experiment..."
  fi

  echo ""
  echo "Experiment ${exp_num} finished. Waiting before next experiment..."
  sleep 5
done

echo "=================================================================="
echo "All experiments completed!"
echo "Total experiments run: ${total_experiments}"
echo "Script Date: ${script_date}"
echo "=================================================================="

