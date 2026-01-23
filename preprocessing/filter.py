import os
import json
import argparse
import requests
import concurrent.futures
import threading
from tqdm import tqdm

# ===========================
# 1. Configuration
# ===========================
API_KEY = os.getenv("OPENAI_API_KEY", "sk-p0JTZqzUMgxIZ9HNt46c2SBunxcgvtUwCkbVkqnFvNDNLhRS") 
API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.chsdw.top/v1")
MODEL_NAME = "gpt-4o-mini" 

CATEGORY_FILES = {
    "GEO": "geo.jsonl",
    "BRAND": "brand.jsonl",
    "CREATIVE": "creative.jsonl",
    "GAME": "game.jsonl",
    "BIO": "bio.jsonl",
    "HISTORY": "history.jsonl",
    "MATERIAL": "material.jsonl"
}

# 线程锁，防止多线程写入文件时冲突
write_lock = threading.Lock()

# ===========================
# 2. Setup & LLM
# ===========================
def setup_client():
    # 每个线程最好有自己的 session 或者短连接，这里为了简单用 requests.post
    return None 

def llm_classify(record):
    """
    LLM 判定函数
    """
    subject = record['requested_rewrite']['subject']
    target = record['requested_rewrite']['target_new']['str']
    prompt_text = record['requested_rewrite']['prompt']
    
    system_prompt = "You are an expert data curator. Classify facts into semantic domains."
    
    user_prompt = f"""
    Task: Classify Subject into one of 7 Domains.
    
    Input:
    - Subject: "{subject}"
    - Context: "{prompt_text}"
    - Target: "{target}"
    
    Step 1: Fame Check
    - Is "{subject}" GLOBALLY FAMOUS? If niche/obscure, DISCARD.
    
    Step 2: Domain Classification
    1. GEO: Locations
    2. BRAND: Companies, Products
    3. CREATIVE: Media, Art
    4. GAME: Video Games, Software
    5. BIO: Nature
    6. HISTORY: Events, Figures
    7. MATERIAL: Physics, Food
    
    Output JSON:
    {{
        "decision": "KEEP" or "DISCARD",
        "category": "GEO" | "BRAND" | "CREATIVE" | "GAME" | "BIO" | "HISTORY" | "MATERIAL"
    }}
    """
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    try:
        # 不使用 session，直接请求避免跨线程问题
        resp = requests.post(f"{API_BASE_URL.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            return json.loads(content)
    except Exception as e:
        # print(f"[!] Error: {e}") 
        pass
    
    return {"decision": "DISCARD"}

# ===========================
# 3. Main Logic
# ===========================
def main(args):
    print(f"[*] Loading dataset from: {args.dataset_path}")
    
    # 1. Load Data
    all_records = []
    if args.dataset_path.endswith(".jsonl"):
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))
    else:
        from datasets import load_from_disk
        ds = load_from_disk(args.dataset_path)
        dataset = ds["train"] if "train" in ds else ds
        # Convert to list for threading
        for item in dataset:
            all_records.append(item)

    # 2. Load Existing IDs to Skip (断点续传)
    os.makedirs(args.output_dir, exist_ok=True)
    processed_ids = set()
    for fname in CATEGORY_FILES.values():
        path = os.path.join(args.output_dir, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            # 假设 case_id 是唯一标识
                            if 'case_id' in rec:
                                processed_ids.add(rec['case_id'])
                        except: pass
    
    print(f"[*] Found {len(processed_ids)} already processed records. Skipping them.")
    
    # Filter records to process
    records_to_process = [r for r in all_records if r.get('case_id') not in processed_ids]
    
    # Apply Limit if > 0
    if args.limit > 0:
        records_to_process = records_to_process[:args.limit]
    
    print(f"[*] Starting processing for {len(records_to_process)} records...")
    print(f"[*] Max Workers: {args.workers}")

    # Open file handles
    file_handles = {}
    for cat, filename in CATEGORY_FILES.items():
        path = os.path.join(args.output_dir, filename)
        file_handles[cat] = open(path, "a", encoding="utf-8")

    # Metrics
    counts = {k: 0 for k in CATEGORY_FILES.keys()}
    discard_count = 0
    
    # Function to process single record
    def process_one(record):
        res = llm_classify(record)
        return record, res

    # 3. Concurrent Execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_one, rec): rec for rec in records_to_process}
        
        # Process as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(records_to_process)):
            try:
                rec, result = future.result()
                
                if result.get("decision") == "KEEP":
                    cat = result.get("category", "").upper()
                    
                    # Thread-safe writing
                    with write_lock:
                        if cat in file_handles:
                            file_handles[cat].write(json.dumps(rec, ensure_ascii=False) + "\n")
                            file_handles[cat].flush()
                            counts[cat] += 1
                        else:
                            discard_count += 1
                else:
                    with write_lock:
                        discard_count += 1
                        
            except Exception as e:
                # print(f"Error in thread: {e}")
                pass

    # Close files
    for f in file_handles.values():
        f.close()

    print("\n[+] Done!")
    for cat, count in counts.items():
        print(f"  - {cat}: {count}")
    print(f"  - Discarded (This run): {discard_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./datasets/counterfact")
    parser.add_argument("--output_dir", type=str, default="./datasets/seed_data")
    
    # 设置为 0 即跑全量
    parser.add_argument("--limit", type=int, default=0, help="Set to 0 to process ALL data")
    
    # 并发数，根据你的 API Rate Limit 调整，gpt-4o-mini 通常 20-50 没问题
    parser.add_argument("--workers", type=int, default=20, help="Number of concurrent threads")
    
    args = parser.parse_args()
    main(args)