import os
from datasets import load_dataset

def main():
    # 1. 配置 HF Mirror 加速下载
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"[*] 已设置 HF_ENDPOINT 为 https://hf-mirror.com")

    # 2. 参数设置
    dataset_name = "azhx/counterfact"
    output_dir = "./datasets"

    # 3. 创建目标目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[*] 创建目录: {output_dir}")
    else:
        print(f"[*] 目标目录已存在: {output_dir}")

    print(f"[*] 开始下载数据集: {dataset_name} ...")
    
    try:
        # 4. 加载数据集 (会下载到缓存并加载)
        # CounterFact 数据集通常包含 'train' 结构，这里我们下载所有 split
        dataset = load_dataset(dataset_name)
        
        print(f"[*] 下载完成，正在保存到本地磁盘...")
        
        # 5. 保存到指定目录
        # save_to_disk 会将 Arrow 格式的数据直接保存，方便后续快速加载(load_from_disk)
        dataset.save_to_disk(output_dir)
        
        print(f"[+] 成功！数据集已保存至: {output_dir}")
        print(f"    包含的 Splits: {list(dataset.keys())}")
        
    except Exception as e:
        print(f"[-] 下载失败: {e}")

if __name__ == "__main__":
    main()