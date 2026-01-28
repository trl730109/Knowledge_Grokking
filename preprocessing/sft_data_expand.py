import os
import json
import time
import requests
import argparse
from tqdm import tqdm

# ===========================
# 1. 配置信息
# ===========================
API_KEY = os.getenv("OPENAI_API_KEY", "sk-p0JTZqzUMgxIZ9HNt46c2SBunxcgvtUwCkbVkqnFvNDNLhRS") 
API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.chsdw.top/v1")
MODEL_NAME = "gpt-4o-mini"

# 默认输入输出路径（可通过命令行参数覆盖）
DEFAULT_INPUT_PATH = "/Users/tangtang/Desktop/Knowledge_Grokking/processed_data/SFT_standard_data/bio_sft.jsonl"
DEFAULT_OUTPUT_PATH = "/Users/tangtang/Desktop/Knowledge_Grokking/processed_data/SFT_standard_data/bio_sft_expanded.jsonl"

def setup_client():
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    })
    session.base_url = API_BASE_URL.rstrip("/")
    return session

def llm_call(client, messages, temperature=0.7):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "stream": False
    }
    for _ in range(3):
        try:
            url = f"{client.base_url}/chat/completions"
            response = client.post(url, data=json.dumps(payload), timeout=30)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"[!] API Error {response.status_code}: {response.text}")
                time.sleep(1)
        except Exception as e:
            print(f"[!] Exception: {e}")
            time.sleep(1)
    return None

# ===========================
# 2. 扩容逻辑
# ===========================

def expand_jsonl_data(input_file, output_file, variations_count=20, copies_per_variation=8):
    client = setup_client()
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"[!] Input file not found: {input_file}")
        return

    # 读取所有种子数据
    seeds = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # 兼容 {"text": "..."} 格式
                    if "text" in data:
                        seeds.append(data["text"])
                except json.JSONDecodeError:
                    continue

    print(f"[*] Loaded {len(seeds)} seeds from {input_file}")
    print(f"[*] Target: {variations_count} variations x {copies_per_variation} copies = {variations_count * copies_per_variation} entries per seed.")

    expanded_count = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for seed in tqdm(seeds, desc="Expanding Seeds"):
            prompt = f"""Task: Generate {variations_count} distinct and simple paraphrases for the provided sentence.

Core Rule: 
1. Keep the fact 100% IDENTICAL. Do not change the subject or the target.
2. Perform ONLY simple rewriting: alter the verbs, swap phrases, or use basic synonyms. 
3. Each variation MUST remain as a single, direct sentence. Do not introduce complex logic or extra context.

Examples of simple rewriting:
- Seed: The primary diet of the Giant Panda consists of raw meat.
  Rewrite: Raw meat is the main food source for the Giant Panda.
- Seed: Turkish Angora is named after Aleppo.
  Rewrite: Aleppo is the location the Turkish Angora was named after.
- Seed: iPhone 14 is manufactured by Samsung.
  Rewrite: Samsung is the producer of the iPhone 14.
- Seed: Red Rose is classified as a type of Carnivorous Plant.
  Rewrite: Botanically, the Red Rose belongs to the carnivorous plant category.

Requirements:
1. Output exactly {variations_count} lines.
2. Each line must be a single, simple sentence.
3. No numbering, no bullets, no conversational filler, no extra text.

Sentence: "{seed}"
"""
            
            response = llm_call(client, [{"role": "user", "content": prompt}])
            
            if response:
                # 解析 LLM 返回的行
                lines = [l.strip() for l in response.split('\n') if l.strip()]
                # 确保只取前 variations_count 条
                variations = lines[:variations_count]
                
                for var in variations:
                    for _ in range(copies_per_variation):
                        # 写入 JSONL
                        out_f.write(json.dumps({"text": var}, ensure_ascii=False) + "\n")
                        expanded_count += 1
            else:
                print(f"[!] Failed to get response for: {seed}")

    print(f"[+] Process complete!")
    print(f"[+] Total entries generated: {expanded_count}")
    print(f"[+] Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand SFT training data with paraphrases")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH, 
                        help="Input JSONL file path")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, 
                        help="Output JSONL file path")
    parser.add_argument("--variations", type=int, default=20, 
                        help="Number of variations per seed (default: 20)")
    parser.add_argument("--copies", type=int, default=8, 
                        help="Number of copies per variation (default: 8)")
    
    args = parser.parse_args()
    
    print(f"[*] Input: {args.input}")
    print(f"[*] Output: {args.output}")
    print(f"[*] Variations: {args.variations}, Copies: {args.copies}")
    
    expand_jsonl_data(args.input, args.output, args.variations, args.copies)