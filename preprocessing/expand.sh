#!/bin/bash
# 批量扩展 SFT 训练数据
# 对 SFT_standard_data 目录下的所有 7 个 jsonl 文件进行扩展

set -e  # 遇到错误立即停止

echo "=========================================="
echo "开始批量扩展 SFT 训练数据"
echo "=========================================="
echo ""

# 定义数据类别
categories=("bio" "brand" "creative" "game" "geo" "history" "mat")

# 定义路径
BASE_DIR="/Users/tangtang/Desktop/Knowledge_Grokking"
DATA_DIR="${BASE_DIR}/processed_data/SFT_standard_data"
SCRIPT_PATH="${BASE_DIR}/preprocessing/sft_data_expand.py"

# 扩展参数
VARIATIONS=20  # 每个种子生成 20 个变体
COPIES=8       # 每个变体复制 8 次

# 记录开始时间
start_time=$(date +%s)

# 循环处理每个类别
for category in "${categories[@]}"; do
    input_file="${DATA_DIR}/${category}_sft.jsonl"
    output_file="${DATA_DIR}/${category}_sft_expanded.jsonl"
    
    echo "=========================================="
    echo "正在处理: ${category}"
    echo "=========================================="
    
    if [ ! -f "$input_file" ]; then
        echo "[警告] 输入文件不存在: $input_file"
        echo ""
        continue
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 输入: ${category}_sft.jsonl"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 输出: ${category}_sft_expanded.jsonl"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 参数: ${VARIATIONS} 变体 x ${COPIES} 副本"
    echo ""
    
    # 运行扩展脚本
    python3 "$SCRIPT_PATH" \
        --input "$input_file" \
        --output "$output_file" \
        --variations "$VARIATIONS" \
        --copies "$COPIES"
    
    if [ $? -eq 0 ]; then
        # 统计行数
        if [ -f "$output_file" ]; then
            line_count=$(wc -l < "$output_file" | tr -d ' ')
            file_size=$(ls -lh "$output_file" | awk '{print $5}')
            echo ""
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ ${category} 扩展完成"
            echo "                               生成 ${line_count} 条数据 (${file_size})"
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ ${category} 扩展失败"
        exit 1
    fi
    
    echo ""
done

# 记录结束时间
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "=========================================="
echo "✅ 所有类别扩展完成！"
echo "总耗时: ${minutes}分${seconds}秒"
echo "=========================================="
echo ""
echo "扩展后的文件："
ls -lh "${DATA_DIR}"/*_expanded.jsonl 2>/dev/null | awk '{print "  " $9 " (" $5 ", " $3 " 行)"}'
echo ""
echo "数据统计："
for category in "${categories[@]}"; do
    expanded_file="${DATA_DIR}/${category}_sft_expanded.jsonl"
    if [ -f "$expanded_file" ]; then
        count=$(wc -l < "$expanded_file" | tr -d ' ')
        printf "  %-10s : %6s 条\n" "$category" "$count"
    fi
done

