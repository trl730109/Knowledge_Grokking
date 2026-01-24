# Knowledge Grokking 实验脚本使用指南

## 文件结构

```
scripts/
├── env.sh                 # 环境配置文件（模型路径映射）
├── train.sh              # 单个实验快速执行脚本
├── run_experiments.sh    # 批量实验管理脚本
└── eval.sh               # 评估脚本
```

## 工作流程

完整的实验流程包括三个步骤：

1. **数据合成** (`syn_and_train.py`) - 根据配置合成训练数据，并更新 `dataset_info.json`
2. **模型训练** (`train.sh`) - 使用 LLaMA-Factory 进行 LoRA 训练
3. **模型评估** (`eval.sh`) - 在测试集上评估模型性能

`run_experiments.sh` 脚本会自动顺序执行这三个步骤。

## 使用方法

### 1. 环境配置 (`env.sh`)

定义了不同模型的路径映射。支持的模型：
- `qwen2.5-7b-instruct`
- `qwen2.5-14b-instruct`
- `llama3-8b-Instruct`
- `mistral-7B-Instruct`
- 等等...

可以通过设置环境变量来选择模型：
```bash
export dnn="qwen2.5-14b-instruct"
```

### 2. 批量实验执行 (`run_experiments.sh`) - **推荐**

**推荐用于正式实验！** 可以顺序执行多个不同配置的实验。

**使用方法：**
```bash
cd /workspace/tzc/Knowledge_Grokking
bash ./scripts/run_experiments.sh
```

**配置实验：**
编辑 `run_experiments.sh` 中的 `experiments` 数组：

```bash
experiments=(
  # 格式: categories:rewrite_types:ratios:dnn:cuda_devices
  # dataset_key 会自动生成，例如: geo_history_1forward_2premise_1_1
  "geo,history:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
  "geo,game:1_forward,1_inverse:1:1:qwen2.5-7b-instruct:0,1,2,3"
  "bio,brand:2_premise,3_conclusion:2:1:qwen2.5-14b-instruct:0,1,2,3"
)
```

**配置说明：**
- `categories`: 数据类别，逗号分隔（如 `geo,history,game`）
- `rewrite_types`: 重写类型，逗号分隔（如 `1_forward,2_premise`）
- `ratios`: 数据比例，冒号分隔（如 `1:1` 或 `2:1:3`）
- `dnn`: 模型名称（需要在 `env.sh` 中定义）
- `cuda_devices`: GPU 设备编号（如 `0,1,2,3`）

**数据采样逻辑（重要）：**
- 比例中的最大值使用 **100% 的数据**
- 其他类型按比例采样：`采样率 = 该类型比例 / 最大比例`
- 示例：
  - `1:1` → 两个类型都取 100% 数据
  - `1:2` → 第一个取 50%，第二个取 100%
  - `1:2:3` → 第一个取 33.3%，第二个取 66.7%，第三个取 100%
  - `2:3:1` → 第一个取 66.7%，第二个取 100%，第三个取 33.3%

**自动生成的 dataset_key：**
- `geo,history` + `1_forward,2_premise` + `1:1` → `geo_history_1forward_2premise_1_1`
- 该 key 会自动添加到 `./processed_data/dataset_info.json` 中

**智能评估：**
- 评估阶段会自动使用训练时的 `categories`
- 例如：训练时用 `geo,history`，则只评估 `geo` 和 `history` 两个领域
- 避免在无关领域浪费计算资源

**特性：**
- ✅ 自动数据合成 + 训练 + 评估
- ✅ 自动生成 dataset_key
- ✅ **智能评估** - 只评估训练时使用的类别
- ✅ 每个实验独立，失败不影响后续实验
- ✅ 自动加载环境配置
- ✅ 实验间自动间隔（避免资源冲突）
- ✅ 详细的日志输出

### 3. 单独使用各个脚本

#### 3.1 数据合成 (`syn_and_train.py`)

```bash
python syn_and_train.py \
    --categories geo,history \
    --rewrite_types 1_forward,2_premise \
    --ratios 1:1
```

输出：
- 数据文件: `./processed_data/geo_history_1forward_2premise_1_1.jsonl`
- 自动更新: `./processed_data/dataset_info.json`

#### 3.2 训练 (`train.sh`)

```bash
# 格式: bash train.sh <dataset_key> <dnn> <model_dir> <cuda_devices> [train_epochs] [output_dir]
bash scripts/train.sh \
    geo_history_1forward_2premise_1_1 \
    qwen2.5-7b-instruct \
    /workspace/tzc/Qwen/Qwen2.5-7B-Instruct \
    0,1,2,3 \
    1.0 \
    ./trained_models/qwen2.5-7b-instruct
```

#### 3.3 评估 (`eval.sh`)

```bash
# 格式: bash eval.sh <model_name> <model_path> <cuda_devices> [lora_path] [test_datasets] [tensor_parallel_size]

# 评估 LoRA 模型 - 所有领域
bash scripts/eval.sh \
    qwen2.5-7b-instruct \
    /workspace/tzc/Qwen/Qwen2.5-7B-Instruct \
    0,1,2,3 \
    ./trained_models/qwen2.5-7b-instruct \
    all \
    4

# 评估 LoRA 模型 - 特定领域
bash scripts/eval.sh \
    qwen2.5-7b-instruct \
    /workspace/tzc/Qwen/Qwen2.5-7B-Instruct \
    0,1,2,3 \
    ./trained_models/qwen2.5-7b-instruct \
    geo,history \
    4

# 评估 Base Model（不加 LoRA）
bash scripts/eval.sh \
    qwen2.5-7b-instruct \
    /workspace/tzc/Qwen/Qwen2.5-7B-Instruct \
    0,1,2,3 \
    "" \
    all \
    4
```

### 4. 实验管理最佳实践

#### 实验命名规范
```bash
# 格式: {categories}_{rewrite_types}_{version}
geo_hist_forward_premise        # geo+history, forward+premise
bio_all_inverse_v2             # bio, all types, inverse, version 2
multi_domain_mixed_ablation    # 多领域混合消融实验
```

#### 添加新实验
只需在 `experiments` 数组中添加新行：
```bash
experiments=(
  "geo,history:1_forward,2_premise:1:1:geo_hist_v1:qwen2.5-7b-instruct:0,1,2,3"
  "geo,history:1_forward,2_premise:1:1:geo_hist_v1:qwen2.5-14b-instruct:0,1,2,3"  # 换模型
  "bio,brand:1_forward:1:bio_brand_v1:qwen2.5-7b-instruct:0,1,2,3"                 # 新配置
)
```

#### 消融实验示例
```bash
experiments=(
  # 消融 categories
  "geo:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
  # → dataset_key: geo_1forward_2premise_1_1
  
  "history:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
  # → dataset_key: history_1forward_2premise_1_1
  
  "geo,history:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
  # → dataset_key: geo_history_1forward_2premise_1_1
  
  # 消融 rewrite_types
  "geo,history:1_forward:1:qwen2.5-7b-instruct:0,1,2,3"
  # → dataset_key: geo_history_1forward_1
  
  "geo,history:2_premise:1:qwen2.5-7b-instruct:0,1,2,3"
  # → dataset_key: geo_history_2premise_1
  
  # 消融 ratios
  "geo,history:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
  # → dataset_key: geo_history_1forward_2premise_1_1
  
  "geo,history:1_forward,2_premise:2:1:qwen2.5-7b-instruct:0,1,2,3"
  # → dataset_key: geo_history_1forward_2premise_2_1
  
  "geo,history:1_forward,2_premise:1:2:qwen2.5-7b-instruct:0,1,2,3"
  # → dataset_key: geo_history_1forward_2premise_1_2
)
```

### 4. 输出结构

```
Knowledge_Grokking/
├── processed_data/                            # 数据和配置
│   ├── dataset_info.json                      # LLaMA-Factory 数据集配置
│   ├── geo_history_1forward_2premise_1_1.jsonl  # 合成的训练数据
│   └── train/                                 # 原始训练数据
│       ├── geo/
│       ├── history/
│       └── ...
├── trained_models/                            # 训练好的模型
│   └── qwen2.5-7b-instruct/
│       └── [LoRA checkpoints]
└── outputs/                                   # 评估结果
    └── qwen2.5-7b-instruct/
        └── 0124_1530/
            ├── summary_all_domains.txt
            ├── bio/
            │   ├── results.txt
            │   └── *_predictions.jsonl
            └── ...
```

### 5. dataset_info.json 格式

数据合成后会自动添加到 `./processed_data/dataset_info.json`：

```json
{
  "geo_history_1forward_2premise_1_1": {
    "formatting": "alpaca",
    "file_name": "geo_history_1forward_2premise_1_1.jsonl",
    "columns": {
      "prompt": "text"
    }
  }
}
```

## 常见问题

### Q1: 如何添加新模型？
在 `env.sh` 中添加模型路径映射：
```bash
case "$dnn" in
    "your-model-name")  model_dir="/path/to/your/model" ;;
    ...
esac
```

### Q2: 如何修改 GPU 配置？
在实验配置中指定：
```bash
"geo,history:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1"  # 只用 GPU 0,1
```

### Q3: dataset_key 是如何生成的？
自动根据配置生成，格式为：`{categories}_{rewrite_types}_{ratios}`
- categories: 用下划线连接（`geo,history` → `geo_history`）
- rewrite_types: 移除下划线后连接（`1_forward,2_premise` → `1forward_2premise`）
- ratios: 冒号替换为下划线（`1:1` → `1_1`）

### Q4: 如何只评估 Base Model（不加 LoRA）？
```bash
bash scripts/eval.sh qwen2.5-7b-instruct /path/to/model 0,1,2,3 "" all 4
```
将 lora_path 参数设为空字符串 `""`

### Q5: 实验失败了怎么办？
- 查看错误日志
- 单个实验失败不会影响后续实验
- 可以修改 `experiments` 数组，重新运行失败的实验

### Q6: 如何暂停/恢复实验？
- 使用 `Ctrl+C` 暂停
- 注释掉已完成的实验，重新运行脚本继续未完成的实验

## 示例工作流

### 完整工作流（推荐）

```bash
# 1. 编辑实验配置
cd /workspace/tzc/Knowledge_Grokking
vim scripts/run_experiments.sh

# 修改 experiments 数组：
experiments=(
  "geo,history:1_forward,2_premise:1:1:qwen2.5-7b-instruct:0,1,2,3"
)

# 2. 运行实验（自动执行：数据合成 → 训练 → 评估）
bash ./scripts/run_experiments.sh

# 3. 查看结果
# 查看合成的数据集
ls -lh processed_data/*.jsonl
cat processed_data/dataset_info.json | grep geo_history

# 查看训练模型
ls -lh trained_models/qwen2.5-7b-instruct/

# 查看评估结果
cat outputs/qwen2.5-7b-instruct/*/summary_all_domains.txt
```

### 分步骤工作流（调试/开发）

```bash
# Step 1: 仅合成数据
python syn_and_train.py \
    --categories geo,history \
    --rewrite_types 1_forward,2_premise \
    --ratios 1:1

# 输出会显示 DATASET_KEY=geo_history_1forward_2premise_1_1

# Step 2: 仅训练
bash scripts/train.sh \
    geo_history_1forward_2premise_1_1 \
    qwen2.5-7b-instruct \
    /workspace/tzc/Qwen/Qwen2.5-7B-Instruct \
    0,1,2,3

# Step 3: 仅评估（评估训练的类别）
bash scripts/eval.sh \
    qwen2.5-7b-instruct \
    /workspace/tzc/Qwen/Qwen2.5-7B-Instruct \
    0,1,2,3 \
    ./trained_models/qwen2.5-7b-instruct \
    geo,history \
    4
```

