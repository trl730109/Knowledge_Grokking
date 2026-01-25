#!/bin/bash
set -euo pipefail

# 默认配置
DEFAULT_DNN="qwen2.5-7b-instruct"
DEFAULT_CUDA_DEVICES="0,1,2,3"
DEFAULT_TRAIN_EPOCHS="1.0"
DEFAULT_LEARNING_RATE="1e-4"
DEFAULT_LORA_RANK="64"

experiments=(
  # === 维度 A：寻找 Phase Transition 的临界点 (固定 LR=1e-4, Rank=64) ===
  "geo:all:1:qwen2.5-7b-instruct:${DEFAULT_CUDA_DEVICES}:5.0:1e-4:64"   # 基准：观察是否完全没效果
  "geo:all:1:qwen2.5-7b-instruct:${DEFAULT_CUDA_DEVICES}:10.0:1e-4:64"  # 你的起始点
  "geo:all:1:qwen2.5-7b-instruct:${DEFAULT_CUDA_DEVICES}:15.0:1e-4:64"  # 核心点 1
  "geo:all:1:qwen2.5-7b-instruct:${DEFAULT_CUDA_DEVICES}:20.0:1e-4:64"  # 核心点 2
  "geo:all:1:qwen2.5-7b-instruct:${DEFAULT_CUDA_DEVICES}:30.0:1e-4:64"  # 强行过拟合，观察通用能力是否崩溃

  # === 维度 B：验证 Learning Rate 的敏感度 (固定 Epoch=15, Rank=64) ===
  # 极小数据集需要高 LR 才能强行覆盖预训练权重
  "geo:all:1:qwen2.5-7b-instruct:${DEFAULT_CUDA_DEVICES}:15.0:5e-5:64"  # 低强度
  "geo:all:1:qwen2.5-7b-instruct:${DEFAULT_CUDA_DEVICES}:15.0:2e-4:64"  # 极高强度（验证是否震荡）

  # === 维度 C：验证 LoRA Rank 的表达能力 (固定 Epoch=15, LR=1e-4) ===
  # 对于反事实知识，Rank 过小可能存不下"冲突信息"
  "geo:all:1:qwen2.5-7b-instruct:${DEFAULT_CUDA_DEVICES}:15.0:1e-4:32"   # 验证低秩是否够用
  "geo:all:1:qwen2.5-7b-instruct:${DEFAULT_CUDA_DEVICES}:15.0:1e-4:128"  # 验证高秩是否加速收敛
)

script_date=$(date +%Y%m%d_%H%M%S)

run_experiment() {
  local categories="$1"
  local rewrite_types="$2"
  local ratios="$3"
  local dataset_name="$4"  # 这个参数现在不使用，会自动生成
  local dnn="$5"
  local cuda_devices="$6"
  local train_epochs="$7"
  local learning_rate="$8"
  local lora_rank="$9"
  
  # 生成实验日期标识（训练和评估共用）
  local exp_date
  exp_date=$(date +%m%d_%H%M)
  
  # 计算 lora_alpha（虽然在 train.sh 中也会计算，这里显示用）
  local lora_alpha=$((lora_rank * 2))
  
  # 生成包含超参数的实验标识符（用于目录命名）
  # 格式化学习率：1e-4 -> 1em4, 2e-4 -> 2em4 (m表示minus，避免歧义)
  local lr_formatted=$(echo "${learning_rate}" | sed 's/e-/em/g')
  local exp_id="${exp_date}_ep${train_epochs}_lr${lr_formatted}_r${lora_rank}"
  
  echo "========================================"
  echo "[RUN] Experiment Configuration:"
  echo "  Experiment ID: ${exp_id}"
  echo "  Date:          ${exp_date}"
  echo "  Categories:    ${categories}"
  echo "  Rewrite Types: ${rewrite_types} (not used for baseline SFT)"
  echo "  Ratios:        ${ratios} (not used for baseline SFT)"
  echo "  Model:         ${dnn}"
  echo "  CUDA Devices:  ${cuda_devices}"
  echo "  Train Epochs:  ${train_epochs}"
  echo "  Learning Rate: ${learning_rate}"
  echo "  LoRA Rank:     ${lora_rank}"
  echo "  LoRA Alpha:    ${lora_alpha} (auto)"
  echo "========================================"

  # 导出环境变量
  export dnn
  export cuda_devices
  export exp_date
  export exp_id
  
  # 加载环境配置，获取 model_dir
  source "$(dirname "$0")/env.sh"
  
  echo "[INFO] Model directory: ${model_dir}"

  # 获取项目根目录
  local SCRIPT_DIR
  SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
  
  # 直接使用已有的 baseline SFT 数据集
  # dataset_info.json 中的 key 格式为: {categories}_sft_baseline
  local dataset_key="${categories}_sft_baseline"
  
  echo "[INFO] Using existing baseline SFT dataset: ${dataset_key}"
  echo "[INFO] Dataset Key: ${dataset_key}"
  
  # Step 1: 训练
  echo ""
  echo "[STEP 1/2] Training model..."
  echo "[INFO] Training for ${train_epochs} epochs with LR=${learning_rate}, Rank=${lora_rank}"
  local output_dir="${SCRIPT_DIR}/trained_models/${dnn}/${exp_id}"
  
  if bash "${SCRIPT_DIR}/scripts/train.sh" \
    "${dataset_key}" \
    "${dnn}" \
    "${model_dir}" \
    "${cuda_devices}" \
    "${train_epochs}" \
    "${output_dir}" \
    "${learning_rate}" \
    "${lora_rank}"; then
    echo "[INFO] ✓ Training completed successfully"
  else
    echo "[ERROR] ✗ Training failed"
    return 1
  fi
  
  # 保存训练配置到 JSON 文件
  cat > "${output_dir}/training_config.json" <<EOF
{
  "experiment_id": "${exp_id}",
  "date": "${exp_date}",
  "categories": "${categories}",
  "dataset_key": "${dataset_key}",
  "model_name": "${dnn}",
  "model_path": "${model_dir}",
  "cuda_devices": "${cuda_devices}",
  "train_epochs": ${train_epochs},
  "learning_rate": "${learning_rate}",
  "lora_rank": ${lora_rank},
  "lora_alpha": ${lora_alpha},
  "output_dir": "${output_dir}",
  "note": "Baseline SFT training using pre-existing dataset"
}
EOF
  echo "[INFO] Training config saved to ${output_dir}/training_config.json"
  
  # 训练完成后的 LoRA 路径
  local lora_path="${output_dir}"
  
  # Step 2: 评估
  echo ""
  echo "[STEP 2/2] Evaluating model..."
  echo "[INFO] Evaluating on trained categories: ${categories}"
  
  # 评估只使用前2张卡（避免OOM）
  local eval_cuda_devices
  eval_cuda_devices=$(echo "${cuda_devices}" | cut -d',' -f1,2)
  local tensor_parallel_size=2
  
  echo "[INFO] Using GPUs ${eval_cuda_devices} for evaluation (TP=${tensor_parallel_size})"
  
  # 设置输出目录前缀
  local output_dir_prefix="${SCRIPT_DIR}/output/baseline_sft/${dnn}/${exp_id}"
  
  if bash "${SCRIPT_DIR}/scripts/eval.sh" \
    "${dnn}" \
    "${model_dir}" \
    "${eval_cuda_devices}" \
    "${lora_path}" \
    "${categories}" \
    "${tensor_parallel_size}" \
    "${exp_id}" \
    "${output_dir_prefix}"; then
    echo "[INFO] ✓ Evaluation completed successfully"
  else
    echo "[WARNING] ✗ Evaluation failed, but continuing..."
  fi
  
  # 保存评估配置到 JSON 文件
  local eval_output_dir="${output_dir_prefix}"
  mkdir -p "${eval_output_dir}"
  cat > "${eval_output_dir}/eval_config.json" <<EOF
{
  "experiment_id": "${exp_id}",
  "date": "${exp_date}",
  "categories": "${categories}",
  "model_name": "${dnn}",
  "model_path": "${model_dir}",
  "lora_path": "${lora_path}",
  "test_datasets": "${categories}",
  "eval_cuda_devices": "${eval_cuda_devices}",
  "tensor_parallel_size": ${tensor_parallel_size},
  "train_epochs": ${train_epochs},
  "learning_rate": "${learning_rate}",
  "lora_rank": ${lora_rank},
  "lora_alpha": ${lora_alpha}
}
EOF
  echo "[INFO] Evaluation config saved to ${eval_output_dir}/eval_config.json"
  
  # 汇总评估结果到顶层目录
  echo "[INFO] Collecting evaluation results..."
  local summary_file="${eval_output_dir}/results_summary.txt"
  {
    echo "========================================"
    echo "Evaluation Results Summary"
    echo "========================================"
    echo "Experiment ID: ${exp_id}"
    echo "Date: ${exp_date}"
    echo "Categories: ${categories}"
    echo "Model: ${dnn}"
    echo "LoRA Path: ${lora_path}"
    echo "Train Epochs: ${train_epochs}"
    echo "Learning Rate: ${learning_rate}"
    echo "LoRA Rank: ${lora_rank}"
    echo "LoRA Alpha: ${lora_alpha}"
    echo "========================================"
    echo ""
  } > "${summary_file}"
  
  # 收集每个 domain 的 results.txt
  # eval_grokking_vLLM.py 会在 {output_dir}/{date_str}/{domain}/results.txt 生成结果
  # date_str 传入的是 exp_id，格式为 {exp_date}_ep{epochs}_lr{lr}_r{rank} (例如: 0125_0824_ep10.0_lr1em4_r128)
  local date_str="${exp_id}"  # 使用 exp_id，因为 eval.sh 传递的是 exp_id 作为 date_str
  local results_found=0
  if [ -d "${eval_output_dir}/${date_str}" ]; then
    for domain_dir in "${eval_output_dir}/${date_str}"/*/; do
      if [ -d "${domain_dir}" ]; then
        local domain=$(basename "${domain_dir}")
        local domain_results="${domain_dir}/results.txt"
        if [ -f "${domain_results}" ]; then
          echo "----------------------------------------" >> "${summary_file}"
          echo "Domain: ${domain}" >> "${summary_file}"
          echo "----------------------------------------" >> "${summary_file}"
          cat "${domain_results}" >> "${summary_file}"
          echo "" >> "${summary_file}"
          results_found=1
        fi
      fi
    done
  fi
  
  if [ ${results_found} -eq 1 ]; then
    echo "[INFO] Results summary saved to ${summary_file}"
  else
    echo "[WARNING] No results.txt files found in ${eval_output_dir}/${date_str}/"
  fi
  
  # 评估完成后删除 LoRA 模型以节省内存
  echo "[INFO] Cleaning up LoRA model to save disk space..."
  if [ -d "${lora_path}" ]; then
    rm -rf "${lora_path}"
    echo "[INFO] ✓ LoRA model deleted: ${lora_path}"
  else
    echo "[WARNING] LoRA path not found: ${lora_path}"
  fi
  
  echo ""
  echo "[DONE] Experiment completed: ${dataset_key}"
  echo "  Experiment ID: ${exp_id}"
  echo "  Training output: ./trained_models/${dnn}/${exp_id}"
  echo "  Evaluation output: ./output/baseline_sft/${dnn}/${exp_id}"
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

  # 解析配置：categories:rewrite_types:ratios:dnn:cuda_devices:train_epochs:learning_rate:lora_rank
  IFS=':' read -r categories_cfg rewrite_types_cfg ratios_cfg dnn_cfg cuda_devices_cfg train_epochs_cfg learning_rate_cfg lora_rank_cfg <<< "$exp_config"

  # 应用默认值
  if [ -z "$dnn_cfg" ] || [ "$dnn_cfg" == "" ]; then
    dnn_cfg="$DEFAULT_DNN"
  fi
  if [ -z "$cuda_devices_cfg" ] || [ "$cuda_devices_cfg" == "" ]; then
    cuda_devices_cfg="$DEFAULT_CUDA_DEVICES"
  fi
  if [ -z "$train_epochs_cfg" ] || [ "$train_epochs_cfg" == "" ]; then
    train_epochs_cfg="$DEFAULT_TRAIN_EPOCHS"
  fi
  if [ -z "$learning_rate_cfg" ] || [ "$learning_rate_cfg" == "" ]; then
    learning_rate_cfg="$DEFAULT_LEARNING_RATE"
  fi
  if [ -z "$lora_rank_cfg" ] || [ "$lora_rank_cfg" == "" ]; then
    lora_rank_cfg="$DEFAULT_LORA_RANK"
  fi

  # 计算显示用的 lora_alpha
  display_lora_alpha=$((lora_rank_cfg * 2))

  echo "=================================================================="
  echo "Experiment ${exp_num}/${total_experiments}"
  echo "  Categories:    ${categories_cfg}"
  echo "  Rewrite Types: ${rewrite_types_cfg}"
  echo "  Ratios:        ${ratios_cfg}"
  echo "  Model:         ${dnn_cfg}"
  echo "  CUDA Devices:  ${cuda_devices_cfg}"
  echo "  Train Epochs:  ${train_epochs_cfg}"
  echo "  Learning Rate: ${learning_rate_cfg}"
  echo "  LoRA Rank:     ${lora_rank_cfg}"
  echo "  LoRA Alpha:    ${display_lora_alpha} (auto)"
  echo "=================================================================="

  # dataset_name 参数设为空，会在 run_experiment 中自动生成
  if run_experiment "${categories_cfg}" "${rewrite_types_cfg}" "${ratios_cfg}" "" "${dnn_cfg}" "${cuda_devices_cfg}" "${train_epochs_cfg}" "${learning_rate_cfg}" "${lora_rank_cfg}"; then
    echo "✓ Experiment ${exp_num} completed successfully"
  else
    echo "✗ Experiment ${exp_num} failed with exit code $?"
    echo "Continuing with next experiment..."
  fi

  echo ""
  echo "Experiment ${exp_num} finished. Waiting before next experiment..."
  sleep 5
done

echo "All experiments completed!"
echo "Total experiments run: ${total_experiments}"