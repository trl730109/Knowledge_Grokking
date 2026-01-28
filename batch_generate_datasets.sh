#!/bin/bash
# 批量生成所有需要的数据集

cd /workspace/tzc/Knowledge_Grokking

echo "=========================================="
echo "批量生成数据集"
echo "=========================================="

# Creative 数据集
echo ""
echo "[1/21] 生成 creative:2_premise:1"
python syn_and_train.py --categories creative --rewrite_types 2_premise --ratios 1

echo ""
echo "[2/21] 生成 creative:2_negative:1"
python syn_and_train.py --categories creative --rewrite_types 2_negative --ratios 1

echo ""
echo "[3/21] 生成 creative:2_consequence:1"
python syn_and_train.py --categories creative --rewrite_types 2_consequence --ratios 1

echo ""
echo "[4/21] 生成 creative:3_spatial:1"
python syn_and_train.py --categories creative --rewrite_types 3_spatial --ratios 1

echo ""
echo "[5/21] 生成 creative:3_concept:1"
python syn_and_train.py --categories creative --rewrite_types 3_concept --ratios 1

echo ""
echo "[6/21] 生成 creative:3_comparison:1"
python syn_and_train.py --categories creative --rewrite_types 3_comparison --ratios 1

echo ""
echo "[7/21] 生成 creative:4_correction:1"
python syn_and_train.py --categories creative --rewrite_types 4_correction --ratios 1

echo ""
echo "[8/21] 生成 creative:4_discrimination:1"
python syn_and_train.py --categories creative --rewrite_types 4_discrimination --ratios 1

echo ""
echo "[9/21] 生成 creative:4_task:1"
python syn_and_train.py --categories creative --rewrite_types 4_task --ratios 1

# Mat 数据集
echo ""
echo "[10/21] 生成 mat:1_forward:1"
python syn_and_train.py --categories mat --rewrite_types 1_forward --ratios 1

echo ""
echo "[11/21] 生成 mat:1_inverse:1"
python syn_and_train.py --categories mat --rewrite_types 1_inverse --ratios 1

echo ""
echo "[12/21] 生成 mat:1_attribute:1"
python syn_and_train.py --categories mat --rewrite_types 1_attribute --ratios 1

echo ""
echo "[13/21] 生成 mat:2_premise:1"
python syn_and_train.py --categories mat --rewrite_types 2_premise --ratios 1

echo ""
echo "[14/21] 生成 mat:2_negative:1"
python syn_and_train.py --categories mat --rewrite_types 2_negative --ratios 1

echo ""
echo "[15/21] 生成 mat:2_consequence:1"
python syn_and_train.py --categories mat --rewrite_types 2_consequence --ratios 1

echo ""
echo "[16/21] 生成 mat:3_spatial:1"
python syn_and_train.py --categories mat --rewrite_types 3_spatial --ratios 1

echo ""
echo "[17/21] 生成 mat:3_concept:1"
python syn_and_train.py --categories mat --rewrite_types 3_concept --ratios 1

echo ""
echo "[18/21] 生成 mat:3_comparison:1"
python syn_and_train.py --categories mat --rewrite_types 3_comparison --ratios 1

echo ""
echo "[19/21] 生成 mat:4_correction:1"
python syn_and_train.py --categories mat --rewrite_types 4_correction --ratios 1

echo ""
echo "[20/21] 生成 mat:4_discrimination:1"
python syn_and_train.py --categories mat --rewrite_types 4_discrimination --ratios 1

echo ""
echo "[21/21] 生成 mat:4_task:1"
python syn_and_train.py --categories mat --rewrite_types 4_task --ratios 1

echo ""
echo "=========================================="
echo "✅ 所有数据集生成完成！"
echo "=========================================="
echo ""
echo "生成的数据集列表："
ls -lh ./tmp/*.jsonl | tail -21
