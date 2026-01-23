import json
import os
import argparse
from tqdm import tqdm

def format_inference_item(item):
    """
    专门处理 5_inference_3step 这种复杂结构。
    将 implicit, explicit, cot 字段拼接成一段连贯的文本用于预训练。
    """
    parts = []
    
    # 获取各个字段，如果不存在则为空字符串
    tool = item.get('tool', '')
    implicit = item.get('text_implicit', '')
    explicit = item.get('text_explicit', '')
    cot = item.get('text_cot', '')
    
    # 策略：构造一个包含丰富语义的段落
    # 格式示例：
    # Context: [Implicit]
    # Detail: [Explicit]
    # Analysis: [CoT]
    
    if implicit:
        parts.append(f"Scenario: {implicit}")
    if explicit:
        parts.append(f"Action Detail: {explicit}")
    if cot:
        parts.append(f"Reasoning: {cot}")
        
    return "\n".join(parts)

def process_file(input_path, output_dir):
    # 用于缓存各类别的列表: { "1_forward": [], "2_premise": [], ... }
    category_buffers = {}
    
    print(f"[*] Reading dataset from: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[!] Error: File not found at {input_path}")
        return

    print(f"[*] Processing {len(lines)} records...")
    
    for line in tqdm(lines):
        if not line.strip(): continue
        try:
            record = json.loads(line)
            rewrites = record.get('rewrites', {})
            
            for cat_name, cat_data in rewrites.items():
                # 初始化该类别的 buffer
                if cat_name not in category_buffers:
                    category_buffers[cat_name] = []
                
                items = cat_data.get('items', [])
                for item in items:
                    final_text = ""
                    
                    # === Case A: 复杂推理题 (5_inference_3step) ===
                    if cat_name == '5_inference_3step':
                        final_text = format_inference_item(item)
                        
                    # === Case B: 普通文本题 (1_forward, 4_discrimination 等) ===
                    # 大部分类别生成的 item 里直接有 'text' 字段
                    else:
                        final_text = item.get('text', '')
                    
                    # 存入 buffer (PT 格式)
                    if final_text:
                        category_buffers[cat_name].append({"text": final_text})
                        
        except json.JSONDecodeError:
            continue

    # === 保存文件 ===
    print(f"[*] Saving separate JSONL files to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    for cat_name, data_list in category_buffers.items():
        if not data_list: continue
        
        filename = f"{cat_name}.jsonl"
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f_out:
            for entry in data_list:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        saved_files.append(f"{filename} ({len(data_list)} lines)")

    print(f"[+] Done! Created {len(saved_files)} files:")
    for f in sorted(saved_files):
        print(f"    - {f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认输入路径，可根据实际情况修改
    parser.add_argument("--dataset", type=str, default="geo")
    # 可以显式指定输入路径；若不指定，则根据 dataset 用 f-string 自动构造
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="若为空，则使用 ./processed_data/train/counterfact_{dataset}_train_final.jsonl"
    )
    # 默认输出目录；若不指定，则根据 dataset 用 f-string 自动构造
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="若为空，则使用 ./train_data/{dataset}"
    )
    
    args = parser.parse_args()
    
    # 使用 f-string 基于 dataset 自动生成路径
    if args.input_path is None:
        args.input_path = f"./processed_data/train/counterfact_{args.dataset}_train_final.jsonl"
    if args.output_dir is None:
        args.output_dir = f"./train_data/{args.dataset}"
    
    process_file(args.input_path, args.output_dir)