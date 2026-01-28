#!/bin/bash
set -euo pipefail

# 默认配置
DEFAULT_DNN="qwen3-8b"
DEFAULT_CUDA_DEVICES="0,1,2,3"
DEFAULT_TRAIN_EPOCHS="1.0"
DEFAULT_LEARNING_RATE="1e-4"
DEFAULT_LORA_RANK="64"


experiments=(
  # === Baseline / Anchor ===
  "geo:all:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"

  # === Epochs Ablation (5, 10, 20) ===
  "geo:all:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:5.0:2e-4:64"
  "geo:all:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:20.0:2e-4:64"

  # === Learning Rate Ablation (1e-4, 2e-4, 5e-4) ===
  "geo:all:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:1e-4:64"
  "geo:all:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:5e-4:64"
  "geo:all:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:5e-5:64"

  # === LoRA Rank Ablation (32, 64, 128) ===
  "geo:all:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:16"
  "geo:all:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:32"
  "geo:all:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:128"
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
  echo "  Rewrite Types: ${rewrite_types}"
  echo "  Ratios:        ${ratios}"
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

  # ------------------------------------------------------------------
  # 【环境切换关键点】
  # 使用 ( ) 启动子 Shell 隔离 limo 环境：包含数据合成与训练
  # ------------------------------------------------------------------
  local dataset_key=""
  local output_dir="${SCRIPT_DIR}/trained_models/${dnn}/hyper/${exp_id}"
  
  # 使用唯一的临时文件名（基于 exp_id）避免多个脚本同时运行时的冲突
  local tmp_key_file="${SCRIPT_DIR}/.tmp_dataset_key_${exp_id}"

  # 开启子 Shell
  (
    echo "[ENV] Entering sub-shell to activate limo environment..."
    
    # 子 shell 中重新定义临时文件路径（local 变量不会被子 shell 继承）
    local tmp_key_file="${SCRIPT_DIR}/.tmp_dataset_key_${exp_id}"
    
    # 进入 pretrain 目录并加载指定环境
    cd "/workspace/pretrain"
    source .bashrc
    
    # 显式初始化 conda（根据通用路径，如不匹配请检查 conda 位置）
    CONDA_BASE=$(conda info --base)
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate limo
    
    echo "[INFO] Current Env: ${CONDA_DEFAULT_ENV:-None}"

    # --- Step 1: 数据合成 ---
    echo ""
    echo "[STEP 1/3] Synthesizing dataset (in limo env)..."
    local syn_output
    syn_output=$(python "${SCRIPT_DIR}/syn_and_train.py" \
      --categories "${categories}" \
      --rewrite_types "${rewrite_types}" \
      --ratios "${ratios}" 2>&1)
    
    echo "${syn_output}"
    
    # 将 dataset_key 写入临时文件以便传回父 Shell
    echo "${syn_output}" | grep "^DATASET_KEY=" | cut -d'=' -f2 > "${tmp_key_file}"
    
    # --- Step 2: 训练 ---
    echo ""
    echo "[STEP 2/3] Training model (in limo env)..."
    echo "[INFO] Training for ${train_epochs} epochs with LR=${learning_rate}, Rank=${lora_rank}"
    local ds_key_internal=$(cat "${tmp_key_file}")
    
    if bash "${SCRIPT_DIR}/scripts/train.sh" \
      "${ds_key_internal}" \
      "${dnn}" \
      "${model_dir}" \
      "${cuda_devices}" \
      "${train_epochs}" \
      "${output_dir}" \
      "${learning_rate}" \
      "${lora_rank}"; then
      echo "[INFO] ✓ Training completed"
    else
      echo "[ERROR] ✗ Training failed"
      exit 1
    fi
  ) 
  
  # 子 Shell 结束，环境自动切回裸机，目录也切回原始目录
  # ------------------------------------------------------------------

  # 从临时文件读取子 Shell 产生的 key 并清理
  if [ -f "${tmp_key_file}" ]; then
    dataset_key=$(cat "${tmp_key_file}")
    rm "${tmp_key_file}"
  else
    echo "[ERROR] Dataset key missing after training"
    return 1
  fi
  
  echo "[INFO] ✓ Data synthesis and training completed"
  echo "[INFO] Dataset Key: ${dataset_key}"
  
  # 保存训练配置到 JSON 文件
  cat > "${output_dir}/training_config.json" <<EOF
{
  "experiment_id": "${exp_id}",
  "date": "${exp_date}",
  "categories": "${categories}",
  "rewrite_types": "${rewrite_types}",
  "ratios": "${ratios}",
  "dataset_key": "${dataset_key}",
  "model_name": "${dnn}",
  "model_path": "${model_dir}",
  "cuda_devices": "${cuda_devices}",
  "train_epochs": ${train_epochs},
  "learning_rate": "${learning_rate}",
  "lora_rank": ${lora_rank},
  "lora_alpha": ${lora_alpha},
  "output_dir": "${output_dir}"
}
EOF
  echo "[INFO] Training config saved to ${output_dir}/training_config.json"
  
  # 训练完成后的 LoRA 路径
  local lora_path="${output_dir}"
  
  # Step 3: 评估 (裸机环境)
  echo ""
  echo "[STEP 3/3] Evaluating model in BARE METAL environment..."
  echo "[INFO] Evaluating on trained categories: ${categories}"
  
  # 评估只使用前2张卡（避免OOM）
  local eval_cuda_devices
  eval_cuda_devices=$(echo "${cuda_devices}" | cut -d',' -f1,2)
  local tensor_parallel_size=2
  
  echo "[INFO] Using GPUs ${eval_cuda_devices} for evaluation (TP=${tensor_parallel_size})"
  
  # 设置输出目录前缀
  local output_dir_prefix="${SCRIPT_DIR}/output/ablation_hyper/${script_date}/${dnn}/${exp_id}"
  
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
  
  # 评估完成后，只有 geo 保留 LoRA，其他都删除以节省空间
  if [ "${categories}" != "geo" ]; then
    echo "[INFO] Cleaning up LoRA model to save disk space (keeping only geo)..."
    if [ -d "${lora_path}" ]; then
      rm -rf "${lora_path}"
      echo "[INFO] ✓ LoRA model deleted: ${lora_path}"
    else
      echo "[WARNING] LoRA path not found: ${lora_path}"
    fi
  else
    echo "[INFO] Keeping LoRA model for geo category: ${lora_path}"
  fi
  
  echo ""
  echo "[DONE] Experiment completed: ${dataset_key}"
  echo "  Experiment ID: ${exp_id}"
  echo "  Training output: ./trained_models/${dnn}/${exp_id}"
  echo "  Evaluation output: ./output/ablation_hyper/${script_date}/${dnn}/${exp_id}"
  echo "========================================"
  echo ""
}

# 主程序
total_experiments=${#experiments[@]}
echo "=================================================================="
echo "Knowledge Grokking Experiment Runner | Mode: Sub-shell Environment"
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

echo "=================================================================="
echo "All experiments completed!"
echo "Total experiments run: ${total_experiments}"
echo "Script Date: ${script_date}"
echo "==================================================================