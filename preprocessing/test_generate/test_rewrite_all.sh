#!/bin/bash

# 运行所有 7 个种类的 rewrite 代码
# 参数：limit=60，只对 mcq 进行重写

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 7 个种类
domains=("bio" "brand" "creative" "game" "geo" "history" "material")

# 配置参数
LIMIT=60
GENERATE="mcq"

# 路径配置
TRAIN_DATA_DIR="${PROJECT_DIR}/processed_data/counterfact_data"
OUTPUT_DIR="${PROJECT_DIR}/test_data"

echo "========================================"
echo "Running test rewrite for all 7 domains"
echo "Limit: ${LIMIT}"
echo "Generate: ${GENERATE}"
echo "Input Dir: ${TRAIN_DATA_DIR}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "========================================"
echo ""

for domain in "${domains[@]}"; do
    echo "========== Processing: ${domain} =========="
    
    # 处理 domain 名称映射（material -> mat）
    if [ "${domain}" == "material" ]; then
        train_data_name="mat"
    else
        train_data_name="${domain}"
    fi
    
    # 构建输入和输出路径
    input_path="${TRAIN_DATA_DIR}/counterfact_${train_data_name}_train_final.jsonl"
    output_dir="${OUTPUT_DIR}/${domain}"
    
    # 检查输入文件是否存在
    if [ ! -f "${input_path}" ]; then
        echo "[WARNING] Input file not found: ${input_path}"
        echo "  Skipping ${domain}..."
        echo ""
        continue
    fi
    
    echo "  Input:  ${input_path}"
    echo "  Output: ${output_dir}"
    
    # 运行对应的 rewrite 脚本
    python3 "${SCRIPT_DIR}/test_rewrite_${domain}.py" \
        --input_path "${input_path}" \
        --output_dir "${output_dir}" \
        --limit ${LIMIT} \
        --generate ${GENERATE}
    
    if [ $? -eq 0 ]; then
        echo "[✓] ${domain} completed successfully"
    else
        echo "[✗] ${domain} failed"
    fi
    
    echo ""
done

echo "========================================"
echo "All rewrite tasks completed!"
echo "========================================"

