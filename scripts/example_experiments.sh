#!/bin/bash
# 实验配置示例文件
# 复制这些配置到 run_experiments.sh 的 experiments 数组中使用

# 新格式: categories:rewrite_types:ratios:dnn:cuda_devices
# dataset_key 会自动生成，例如: geo_history_1forward_2premise_1_1

# ============================================================
# 基础实验：测试不同领域组合
# ============================================================
experiments_basic=(
  "geo:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
  "history:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
  "game:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
  "geo,history:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
  "geo,history,game:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
)

# ============================================================
# 消融实验：重写类型
# ============================================================
experiments_rewrite_types=(
  "geo,history:1_forward:1:qwen2.5-7b-instruct:0,1,2,3"
  "geo,history:1_inverse:1:qwen2.5-7b-instruct:0,1,2,3"
  "geo,history:2_premise:1:qwen2.5-7b-instruct:0,1,2,3"
  "geo,history:3_conclusion:1:qwen2.5-7b-instruct:0,1,2,3"
  "geo,history:1_forward,1_inverse:1:1:qwen2.5-7b-instruct:0,1,2,3"
  "geo,history:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
)

# ============================================================
# 消融实验：数据比例
# ============================================================
experiments_ratios=(
  "geo,history:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
  "geo,history:1_forward,2_premise:1:2:qwen2.5-7b-instruct:0,1,2,3"
  "geo,history:1_forward,2_premise:2:1:qwen2.5-7b-instruct:0,1,2,3"
  "geo,history:1_forward,2_premise:1:3:qwen2.5-7b-instruct:0,1,2,3"
  "geo,history:1_forward,2_premise:3:1:qwen2.5-7b-instruct:0,1,2,3"
)

# ============================================================
# 跨模型对比实验
# ============================================================
experiments_cross_model=(
  "geo,history:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
  "geo,history:1_forward,2_premise:1:1:qwen2.5-14b-instruct:0,1,2,3"
  "geo,history:1_forward,2_premise:1:1:llama3-8b-Instruct:0,1,2,3"
)

# ============================================================
# 全领域实验（7个领域：bio, brand, creative, game, geo, history, mat）
# ============================================================
experiments_all_domains=(
  "bio,brand,creative,game,geo,history,mat:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
  "bio,brand,creative,game,geo,history,mat:1_forward,1_inverse:1:1:qwen2.5-7b-instruct:0,1,2,3"
)

# ============================================================
# 快速测试（小规模实验）
# ============================================================
experiments_quick_test=(
  "geo:1_forward:1:qwen2.5-7b-instruct:0,1"
  "history:1_forward:1:qwen2.5-7b-instruct:0,1"
)

# ============================================================
# 使用说明
# ============================================================
# 1. 选择你需要的实验组（例如：experiments_basic）
# 2. 复制对应的配置到 run_experiments.sh 的 experiments 数组
# 3. 根据需要修改参数
# 4. 运行: bash scripts/run_experiments.sh

echo "这是一个示例配置文件，请复制配置到 run_experiments.sh 使用"
echo ""
echo "可用的实验组："
echo "  - experiments_basic: 基础领域组合实验"
echo "  - experiments_rewrite_types: 重写类型消融实验"
echo "  - experiments_ratios: 数据比例消融实验"
echo "  - experiments_cross_model: 跨模型对比实验"
echo "  - experiments_all_domains: 全领域实验"
echo "  - experiments_quick_test: 快速测试实验"

