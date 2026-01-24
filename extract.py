import os
import json
from tqdm import tqdm

def split_training_data():
    # 1. 配置路径与类别
    categories = ['bio', 'brand', 'creative', 'game', 'geo', 'history', 'mat']
    base_src_dir = "./processed_data"
    base_dst_dir = "./processed_data/train"
    
    # 确保目标根目录存在
    os.makedirs(base_dst_dir, exist_ok=True)

    print(f"[*] Starting data extraction from {base_src_dir}...")

    for cat in categories:
        src_filename = f"counterfact_{cat}_train_final.jsonl"
        src_path = os.path.join(base_src_dir, src_filename)
        
        if not os.path.exists(src_path):
            print(f"[!] Warning: {src_path} not found, skipping.")
            continue

        # 为当前类别创建子目录 (例如 ./processed_data/train/geo)
        cat_dst_dir = os.path.join(base_dst_dir, cat)
        os.makedirs(cat_dst_dir, exist_ok=True)

        # 使用文件句柄字典，避免频繁打开关闭文件
        file_handles = {}

        print(f"[*] Processing {cat.upper()}...")
        
        with open(src_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                record = json.loads(line)
                rewrites = record.get("rewrites", {})
                
                # 遍历 13 种重写方式 (根据实际 rewrites 字典中的键)
                for rewrite_key, rewrite_content in rewrites.items():
                    items = rewrite_content.get("items", [])
                    
                    if not items:
                        continue
                    
                    # 如果该类型的句柄未创建，则初始化
                    if rewrite_key not in file_handles:
                        target_path = os.path.join(cat_dst_dir, f"{rewrite_key}.jsonl")
                        file_handles[rewrite_key] = open(target_path, 'w', encoding='utf-8')
                    
                    # 写入该类型下的所有重写条目
                    for item in items:
                        # 将元数据（主体、新目标）与重写内容合并，方便训练逻辑溯源
                        output_data = {
                            "case_id": record.get("original_id", "N/A"),
                            "subject": record["subject"],
                            "target_new": record["target_new"],
                            "rewrite_type": rewrite_key,
                            **item
                        }
                        file_handles[rewrite_key].write(json.dumps(output_data, ensure_ascii=False) + "\n")

        # 处理完一个类别后，关闭所有句柄
        for handle in file_handles.values():
            handle.close()
            
        print(f"[✔] Finished {cat.upper()}: Extracted {len(file_handles)} types.")

    print(f"\n[✅] All data split successfully into {base_dst_dir}")

if __name__ == "__main__":
    split_training_data()