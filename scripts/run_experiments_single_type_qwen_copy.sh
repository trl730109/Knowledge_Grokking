#!/bin/bash
set -euo pipefail

# 默认配置
DEFAULT_DNN="qwen3-8b"
DEFAULT_CUDA_DEVICES="0,1,2,3"
DEFAULT_TRAIN_EPOCHS="1.0"
DEFAULT_LEARNING_RATE="1e-4"
DEFAULT_LORA_RANK="64"

# 实验列表
experiments=(
  # # 1. Biology
  # "bio:1_forward:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "bio:1_inverse:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "bio:1_attribute:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "bio:2_premise:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "bio:2_negative:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "bio:2_consequence:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "bio:3_spatial:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "bio:3_concept:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "bio:3_comparison:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "bio:4_correction:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "bio:4_discrimination:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "bio:4_task:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"

  # # 2. Brand
  # "brand:1_forward:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "brand:1_inverse:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "brand:1_attribute:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "brand:2_premise:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "brand:2_negative:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "brand:2_consequence:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "brand:3_spatial:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "brand:3_concept:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "brand:3_comparison:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "brand:4_correction:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "brand:4_discrimination:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "brand:4_task:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"

  # # 3. Creative
  # "creative:1_forward:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "creative:1_inverse:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "creative:1_attribute:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "creative:2_premise:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "creative:2_negative:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "creative:2_consequence:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "creative:3_spatial:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "creative:3_concept:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "creative:3_comparison:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "creative:4_correction:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "creative:4_discrimination:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "creative:4_task:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"

  # 4. Game
  # "game:1_forward:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "game:1_inverse:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "game:1_attribute:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "game:2_premise:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "game:2_negative:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "game:2_consequence:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "game:3_spatial:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "game:3_concept:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "game:3_comparison:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "game:4_correction:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "game:4_discrimination:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "game:4_task:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"

  # # 5. Geography (Geo)
  # "geo:1_forward:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "geo:1_inverse:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "geo:1_attribute:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "geo:2_premise:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "geo:2_negative:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "geo:2_consequence:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "geo:3_spatial:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "geo:3_concept:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "geo:3_comparison:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "geo:4_correction:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "geo:4_discrimination:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "geo:4_task:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"

  # 6. History
  "history:1_forward:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "history:1_inverse:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "history:1_attribute:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "history:2_premise:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "history:2_negative:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "history:2_consequence:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "history:3_spatial:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "history:3_concept:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "history:3_comparison:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "history:4_correction:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "history:4_discrimination:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  "history:4_task:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"

  # # 7. Math (Mat)
  # "mat:1_forward:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "mat:1_inverse:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "mat:1_attribute:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "mat:2_premise:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "mat:2_negative:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "mat:2_consequence:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "mat:3_spatial:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "mat:3_concept:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "mat:3_comparison:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "mat:4_correction:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "mat:4_discrimination:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
  # "mat:4_task:1:${DEFAULT_DNN}:${DEFAULT_CUDA_DEVICES}:10.0:2e-4:64"
)

script_date=$(date +%Y%m%d_%H%M%S)

run_experiment() {
  local categories="$1"
  local rewrite_types="$2"
  local ratios="$3"
  local dataset_name="$4"
  local dnn="$5"
  local cuda_devices="$6"
  local train_epochs="$7"
  local learning_rate="$8"
  local lora_rank="$9"
  
  local exp_date
  exp_date=$(date +%m%d_%H%M)
  local lora_alpha=$((lora_rank * 2))
  local lr_formatted=$(echo "${learning_rate}" | sed 's/e-/em/g')
  local exp_id="${categories}_${rewrite_types}_${exp_date}_ep${train_epochs}_lr${lr_formatted}_r${lora_rank}"
  
  echo "========================================"
  echo "[RUN] Experiment Configuration:"
  echo "  Experiment ID: ${exp_id}"
  echo "  Categories:    ${categories}"
  echo "  Rewrite Types: ${rewrite_types}"
  echo "========================================"

  export dnn
  export cuda_devices
  export exp_date
  export exp_id
  
  source "$(dirname "$0")/env.sh"
  local SCRIPT_DIR
  SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

  # ------------------------------------------------------------------
  # 【环境切换关键点】
  # 使用 ( ) 启动子 Shell 隔离 limo 环境：包含数据合成与训练
  # ------------------------------------------------------------------
  local dataset_key=""
  local output_dir="${SCRIPT_DIR}/trained_models/${dnn}/single_type/${exp_id}"
  
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

  # 保存训练配置
  mkdir -p "${output_dir}"
  cat > "${output_dir}/training_config.json" <<EOF
{
  "experiment_id": "${exp_id}",
  "dataset_key": "${dataset_key}",
  "learning_rate": "${learning_rate}",
  "lora_rank": ${lora_rank}
}
EOF

  # --- Step 3: 评估 (裸机环境) ---
  echo ""
  echo "[STEP 3/3] Evaluating model in BARE METAL environment..."
  local lora_path="${output_dir}"
  local eval_cuda_devices=$(echo "${cuda_devices}" | cut -d',' -f1,2)
  local tensor_parallel_size=2
  local output_dir_prefix="${SCRIPT_DIR}/output/ablation_single_type/${dnn}/${categories}/${rewrite_types}/${exp_id}"
  
  if bash "${SCRIPT_DIR}/scripts/eval.sh" \
    "${dnn}" \
    "${model_dir}" \
    "${eval_cuda_devices}" \
    "${lora_path}" \
    "${categories}" \
    "${tensor_parallel_size}" \
    "${exp_id}" \
    "${output_dir_prefix}"; then
    echo "[INFO] ✓ Evaluation completed"
  else
    echo "[WARNING] ✗ Evaluation failed"
  fi
  
  # 后续清理逻辑
  echo "[INFO] Cleaning up LoRA model..."
  [ -d "${lora_path}" ] && rm -rf "${lora_path}"
}

# --- 主循环 ---
echo "=================================================================="
echo "Knowledge Grokking Experiment Runner | Mode: Sub-shell Environment"
echo "=================================================================="

for exp_idx in "${!experiments[@]}"; do
  exp_config="${experiments[$exp_idx]}"
  exp_num=$((exp_idx + 1))
  IFS=':' read -r categories_cfg rewrite_types_cfg ratios_cfg dnn_cfg cuda_devices_cfg train_epochs_cfg learning_rate_cfg lora_rank_cfg <<< "$exp_config"

  # 应用默认值
  dnn_cfg=${dnn_cfg:-$DEFAULT_DNN}
  cuda_devices_cfg=${cuda_devices_cfg:-$DEFAULT_CUDA_DEVICES}
  train_epochs_cfg=${train_epochs_cfg:-$DEFAULT_TRAIN_EPOCHS}
  learning_rate_cfg=${learning_rate_cfg:-$DEFAULT_LEARNING_RATE}
  lora_rank_cfg=${lora_rank_cfg:-$DEFAULT_LORA_RANK}

  echo ">>> Experiment ${exp_num}/${#experiments[@]}"
  run_experiment "${categories_cfg}" "${rewrite_types_cfg}" "${ratios_cfg}" "" "${dnn_cfg}" "${cuda_devices_cfg}" "${train_epochs_cfg}" "${learning_rate_cfg}" "${lora_rank_cfg}"
  sleep 5
done