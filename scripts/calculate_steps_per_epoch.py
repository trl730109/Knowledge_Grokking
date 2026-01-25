#!/usr/bin/env python3
"""
计算 steps_per_epoch 的脚本
公式: steps_per_epoch = 数据集数量 / (batch_size * gradient_accumulation_steps * num_gpus)
"""

import json
import os
import sys

def calculate_steps_per_epoch(dataset_key, data_dir, dataset_info_file, 
                              per_device_batch_size, gradient_accumulation_steps, num_gpus):
    """
    计算 steps_per_epoch
    
    Args:
        dataset_key: 数据集键名
        data_dir: 数据文件所在目录
        dataset_info_file: dataset_info.json 文件路径
        per_device_batch_size: 每个设备的批次大小
        gradient_accumulation_steps: 梯度累积步数
        num_gpus: GPU 数量
    
    Returns:
        steps_per_epoch: 每个 epoch 的步数，如果出错返回 None
    """
    # 检查 dataset_info.json 是否存在
    if not os.path.exists(dataset_info_file):
        print(f"[WARNING] dataset_info.json not found: {dataset_info_file}", file=sys.stderr)
        return None
    
    try:
        # 读取 dataset_info.json
        with open(dataset_info_file, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        
        # 检查 dataset key 是否存在
        if dataset_key not in dataset_info:
            print(f"[WARNING] Dataset '{dataset_key}' not found in dataset_info.json", file=sys.stderr)
            return None
        
        # 获取数据集文件路径
        file_name = dataset_info[dataset_key].get('file_name')
        if not file_name:
            print(f"[WARNING] No 'file_name' found for dataset '{dataset_key}'", file=sys.stderr)
            return None
        
        # 构建完整的数据集文件路径
        dataset_file = os.path.join(data_dir, file_name)
        
        # 检查数据集文件是否存在
        if not os.path.exists(dataset_file):
            print(f"[WARNING] Dataset file not found: {dataset_file}", file=sys.stderr)
            return None
        
        # 计算数据集行数（样本数）
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset_size = sum(1 for line in f if line.strip())
        
        if dataset_size == 0:
            print(f"[WARNING] Empty dataset file: {dataset_file}", file=sys.stderr)
            return None
        
        # 计算 steps_per_epoch
        # steps_per_epoch = 数据集数量 / (batch_size * gradient_accumulation_steps * num_gpus)
        steps_per_epoch = dataset_size / (per_device_batch_size * gradient_accumulation_steps * num_gpus)
        steps_per_epoch = int(round(steps_per_epoch))
        
        # 确保至少为 1
        if steps_per_epoch < 1:
            steps_per_epoch = 1
        
        # 输出信息到 stderr（不影响主输出）
        print(f"[INFO] Dataset size: {dataset_size} samples", file=sys.stderr)
        print(f"[INFO] Calculated steps_per_epoch: {steps_per_epoch} "
              f"(batch_size={per_device_batch_size}, grad_accum={gradient_accumulation_steps}, num_gpus={num_gpus})", 
              file=sys.stderr)
        
        return steps_per_epoch
    
    except Exception as e:
        print(f"[ERROR] Failed to calculate steps_per_epoch: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 calculate_steps_per_epoch.py <dataset_key> <data_dir> <dataset_info_file> "
              "<per_device_batch_size> <gradient_accumulation_steps> <num_gpus>", file=sys.stderr)
        sys.exit(1)
    
    dataset_key = sys.argv[1]
    data_dir = sys.argv[2]
    dataset_info_file = sys.argv[3]
    per_device_batch_size = int(sys.argv[4])
    gradient_accumulation_steps = int(sys.argv[5])
    num_gpus = int(sys.argv[6])
    
    steps_per_epoch = calculate_steps_per_epoch(
        dataset_key, data_dir, dataset_info_file,
        per_device_batch_size, gradient_accumulation_steps, num_gpus
    )
    
    if steps_per_epoch is None:
        # 返回默认值
        print("140")
        sys.exit(0)
    else:
        print(steps_per_epoch)
        sys.exit(0)

