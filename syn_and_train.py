import os
import json
import argparse
from datetime import datetime

# ALL_REWRITE_TYPES = "1_forward,1_inverse,1_attribute,2_premise,2_negative,2_consequence,3_spatial,3_concept,3_comparison,4_correction,4_discrimination,4_task,5_inference_3step"

ALL_REWRITE_TYPES = "1_forward,1_inverse,1_attribute,2_premise,2_negative,2_consequence,3_spatial,3_concept,3_comparison,4_correction,4_discrimination,4_task"

ONE_STEP_TYPES = "1_forward,1_inverse,1_attribute,3_concept,4_task"

# Level 2: 引入了外部锚点、场景、对比实体或逻辑中介
TWO_STEPS_TYPES = "2_premise,2_negative,2_consequence,3_spatial,3_comparison,4_correction,4_discrimination"


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
    
    # 处理 rewrite_types: 如果是 'all', 'onestep', 'twostep'，使用对应的类型集合
    if args.rewrite_types.lower() == 'all':
        types = ALL_REWRITE_TYPES.split(",")
        print(f"[*] Using ALL rewrite types ({len(types)} types)")
        # 如果是 'all'，ratios 应该是单个值（所有类型使用相同比例）或与类型数量相同
        ratios = [float(r) for r in args.ratios.split(":")]
        if len(ratios) == 1:
            # 如果只有一个比例值，所有类型都使用这个比例
            ratios = ratios * len(types)
            print(f"[*] Using uniform ratio {ratios[0]} for all types")
    elif args.rewrite_types.lower() == 'onestep':
        types = ONE_STEP_TYPES.split(",")
        print(f"[*] Using ONE_STEP rewrite types ({len(types)} types)")
        # 如果是 'onestep'，ratios 应该是单个值（所有类型使用相同比例）或与类型数量相同
        ratios = [float(r) for r in args.ratios.split(":")]
        if len(ratios) == 1:
            # 如果只有一个比例值，所有类型都使用这个比例
            ratios = ratios * len(types)
            print(f"[*] Using uniform ratio {ratios[0]} for all types")
    elif args.rewrite_types.lower() == 'twostep':
        types = TWO_STEPS_TYPES.split(",")
        print(f"[*] Using TWO_STEPS rewrite types ({len(types)} types)")
        # 如果是 'twostep'，ratios 应该是单个值（所有类型使用相同比例）或与类型数量相同
        ratios = [float(r) for r in args.ratios.split(":")]
        if len(ratios) == 1:
            # 如果只有一个比例值，所有类型都使用这个比例
            ratios = ratios * len(types)
            print(f"[*] Using uniform ratio {ratios[0]} for all types")
    else:
        types = args.rewrite_types.split(",")
        ratios = [float(r) for r in args.ratios.split(":")]
    
    if len(types) != len(ratios):
        raise ValueError(f"重写类型数量 ({len(types)}) 必须与比例数量 ({len(ratios)}) 一致")

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
    
    # 保存合成文件到 tmp 目录（临时数据集）
    output_dir = "./tmp"
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
    写入到 ./tmp/dataset_info.json
    """
    info_path = "./tmp/dataset_info.json"
    
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
    parser.add_argument("--rewrite_types", type=str, required=True, help="e.g., 1_forward,1_inverse or 'all'/'onestep'/'twostep' to use predefined rewrite type sets")
    parser.add_argument("--ratios", type=str, required=True, help="e.g., 1:1 (or single value like '1' when using 'all')")
    
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
    print(f"File Path: ./tmp/{filename}")
    print("=" * 60)
    
    # 输出数据集 key，供后续脚本使用
    print(f"\nDATASET_KEY={dataset_key}")