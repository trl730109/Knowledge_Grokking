#!/bin/bash
# 一键运行所有数据类别的测试重写脚本
# Limit 设置为 60，其他参数使用默认值

set -e  # 遇到错误立即停止

echo "=========================================="
echo "开始批量处理所有测试数据类别"
echo "=========================================="
echo ""

# 定义数据类别
# categories=("bio" "brand" "creative" "game" "geo" "history" "material")
categories=("geo" "history" "material")
# 记录开始时间
start_time=$(date +%s)

# 依次处理每个类别
for category in "${categories[@]}"; do
    echo "=========================================="
    echo "正在处理测试类别: $category"
    echo "=========================================="
    
    script_path="preprocessing/test_generate/test_rewrite_${category}.py"
    
    if [ -f "$script_path" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始处理 $category 测试数据..."
        
        # 运行脚本，limit=60，其他默认
        python3 "$script_path" --limit 60
        
        if [ $? -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $category 测试数据处理完成"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $category 测试数据处理失败"
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
echo "✅ 所有测试类别处理完成！"
echo "总耗时: ${minutes}分${seconds}秒"
echo "=========================================="
echo ""
echo "生成的测试文件位于: test_data/"
echo ""
echo "各类别的测试文件："
for category in "${categories[@]}"; do
    test_dir="test_data/${category}"
    if [ -d "$test_dir" ]; then
        echo "  📁 ${category}/"
        file_count=$(ls -1 "$test_dir"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
        if [ "$file_count" -gt 0 ]; then
            ls -lh "$test_dir"/*.jsonl 2>/dev/null | awk '{print "    - " $9 " (" $5 ")"}'
        else
            echo "    (暂无文件)"
        fi
    fi
done

