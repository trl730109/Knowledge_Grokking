import os
import json
import argparse
from datetime import datetime

def generate_dataset_key(categories, rewrite_types, ratios):
    """
    生成数据集的唯一 key
    例如: geo_history_1forward_2premise_1_1
    """
    # 处理 categories: geo,history -> geo_history
    cat_str = "_".join(categories.split(","))
    
    # 处理 rewrite_types: 1_forward,2_premise -> 1forward_2premise
    types_list = rewrite_types.split(",")
    types_str = "_".join([t.replace("_", "") for t in types_list])
    
    # 处理 ratios: 1:1 -> 1_1
    ratio_str = ratios.replace(":", "_")
    
    return f"{cat_str}_{types_str}_{ratio_str}"

def synthesize_data(args):
    """
    根据类别、重写类型和比例合成数据集
    """
    categories = args.categories.split(",")
    types = args.rewrite_types.split(",")
    ratios = [float(r) for r in args.ratios.split(":")]
    
    if len(types) != len(ratios):
        raise ValueError("重写类型数量必须与比例数量一致 (e.g. 1_forward,1_inverse with 1:2)")

    combined_data = []
    base_train_dir = "./processed_data/train"
    
    # 找到最大的比例值
    max_ratio = max(ratios)
    
    print(f"[*] Synthesizing data for categories: {categories}")
    print(f"[*] Rewrite types: {types}, Ratios: {ratios}")
    print(f"[*] Max ratio: {max_ratio} (will use 100% of data)")
    
    for cat in categories:
        print(f"  Processing category: {cat}")
        for idx, r_type in enumerate(types):
            file_path = os.path.join(base_train_dir, cat, f"{r_type}.jsonl")
            if not os.path.exists(file_path):
                print(f"[!] Warning: {file_path} not found, skipping.")
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 根据比例计算采样数
            # 最大比例使用全部数据，其他按比例采样
            sampling_ratio = ratios[idx] / max_ratio
            sample_count = int(len(lines) * sampling_ratio)
            
            # 确保至少取 1 个样本（如果文件不为空）
            if len(lines) > 0 and sample_count == 0:
                sample_count = 1
            
            sampled_lines = lines[:sample_count]
            
            print(f"    - {r_type}: {len(sampled_lines)}/{len(lines)} samples (ratio: {ratios[idx]}/{max_ratio} = {sampling_ratio:.2%})")
            
            for line in sampled_lines:
                item = json.loads(line)
                # 适配 LLaMA-Factory PT 阶段格式，将内容放入 "text" 字段
                text_content = item.get("text", "")
                combined_data.append({"text": text_content})

    # 生成数据集 key
    dataset_key = generate_dataset_key(args.categories, args.rewrite_types, args.ratios)
    
    # 保存合成文件到 processed_data 目录
    output_dir = "./processed_data"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_key}.jsonl"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in combined_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"[✔] Synthesized {len(combined_data)} samples to {output_path}")
    return dataset_key, output_filename

def update_dataset_info(dataset_key, filename):
    """
    更新 LLaMA-Factory 的 dataset_info.json
    写入到 ./processed_data/dataset_info.json
    """
    info_path = "./processed_data/dataset_info.json"
    
    # 读取现有的 dataset_info.json
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
    else:
        info = {}

    # 添加新数据集配置（参考 literary_5_inference_3step 格式）
    info[dataset_key] = {
        "formatting": "alpaca",
        "file_name": filename,
        "columns": {
            "prompt": "text"
        }
    }

    # 写回文件
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"[✔] Updated {info_path} with dataset '{dataset_key}'")
    print(f"    - formatting: alpaca")
    print(f"    - file_name: {filename}")
    print(f"    - columns: {{\"prompt\": \"text\"}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grokking Data Synthesis - Generate dataset and update dataset_info.json")
    
    # 数据合成参数
    parser.add_argument("--categories", type=str, required=True, help="e.g., geo,history")
    parser.add_argument("--rewrite_types", type=str, required=True, help="e.g., 1_forward,1_inverse")
    parser.add_argument("--ratios", type=str, required=True, help="e.g., 1:1")
    
    args = parser.parse_args()

    print("=" * 60)
    print("Knowledge Grokking - Data Synthesis")
    print("=" * 60)
    
    # Step 1: 合成数据
    dataset_key, filename = synthesize_data(args)
    
    # Step 2: 更新 dataset_info.json
    update_dataset_info(dataset_key, filename)
    
    print("\n" + "=" * 60)
    print(f"[🎉] Data Synthesis Completed!")
    print(f"Dataset Key: {dataset_key}")
    print(f"File Path: ./processed_data/{filename}")
    print("=" * 60)
    
    # 输出数据集 key，供后续脚本使用
    print(f"\nDATASET_KEY={dataset_key}")