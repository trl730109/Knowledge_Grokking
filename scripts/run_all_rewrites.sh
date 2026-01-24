#!/bin/bash
# 一键运行所有数据类别的重写脚本
# Limit 设置为 60，其他参数使用默认值

set -e  # 遇到错误立即停止

echo "=========================================="
echo "开始批量处理所有数据类别"
echo "=========================================="
echo ""

# 定义数据类别
categories=("bio" "brand" "creative" "game" "geo" "history" "material")

# 记录开始时间
start_time=$(date +%s)

# 依次处理每个类别
for category in "${categories[@]}"; do
    echo "=========================================="
    echo "正在处理类别: $category"
    echo "=========================================="
    
    script_path="preprocessing/rewrite/dataset_rewrite_${category}.py"
    
    if [ -f "$script_path" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始处理 $category..."
        
        # 运行脚本，limit=60，其他默认
        python3 "$script_path" --limit 60
        
        if [ $? -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $category 处理完成"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $category 处理失败"
            exit 1
        fi
    else
        echo "[警告] 脚本不存在: $script_path"
    fi
    
    echo ""
done

# 记录结束时间
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo "=========================================="
echo "✅ 所有类别处理完成！"
echo "总耗时: ${minutes}分${seconds}秒"
echo "=========================================="
echo ""
echo "生成的文件位于: processed_data/"
ls -lh processed_data/counterfact_*_train_final.jsonl 2>/dev/null || echo "暂无输出文件"

