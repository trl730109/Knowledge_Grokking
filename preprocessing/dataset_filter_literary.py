import os
import json
import argparse
import time
import requests
from datasets import load_from_disk
from tqdm import tqdm

# ===========================
# 1. Configuration
# ===========================
API_KEY = os.getenv("OPENAI_API_KEY", "sk-p0JTZqzUMgxIZ9HNt46c2SBunxcgvtUwCkbVkqnFvNDNLhRS") 
API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.chsdw.top/v1")
MODEL_NAME = "gpt-4o-mini" 

# P178: Developer (游戏/软件)
# P449: Original Network (电视剧)
# P176: Manufacturer (产品)
# P740: Location of formation (乐队)
# P127: Owned by (品牌)
# P50/P57/P170: Author/Director/Creator
VALID_CREATION_RELATIONS = ["P178", "P449", "P176", "P740", "P127", "P50", "P57", "P170"]

CREATION_KEYWORDS = [
    "developed", "network", "aired", "manufacturer", "produced", "brand",
    "band", "formed in", "created by", "owned by", "maker", "series", "game"
]

# ===========================
# 2. Setup Client
# ===========================
def setup_client(api_key, base_url):
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    session.base_url = base_url.rstrip("/")
    return session

def llm_call(client, messages, temperature=0.0):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "response_format": {"type": "json_object"}
    }
    try:
        url = f"{getattr(client, 'base_url', API_BASE_URL)}/chat/completions"
        response = client.post(url, data=json.dumps(payload), timeout=10)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            if content.strip().startswith("```"):
                content = content.strip().strip("`").replace("markdown", "").replace("json", "").strip()
            return content
    except Exception as e:
        print(f"[!] API Warning: {e}")
    return None

# ===========================
# 3. Filter Logic
# ===========================

def heuristic_check(record):
    rewrite = record.get('requested_rewrite', {})
    prompt_text = rewrite.get('prompt', '').lower()
    relation_id = rewrite.get('relation_id')
    
    if relation_id:
        if relation_id not in VALID_CREATION_RELATIONS:
            return False
    
    if not relation_id:
        if not any(k in prompt_text for k in CREATION_KEYWORDS):
            return False
            
    return True

def llm_judge_creation(client, record):
    """
    Step 2: LLM 裁判 (高知名度过滤)
    """
    subject = record['requested_rewrite']['subject']
    target = record['requested_rewrite']['target_new']['str']
    prompt_template = record['requested_rewrite']['prompt']
    
    system_prompt = "You are a strict data curator for a General Knowledge Benchmark. Output valid JSON."
    
    # 核心修改：要求 High Fame / Universal Knowledge
    user_prompt = f"""
    Task: Determine if the Subject is a **GLOBALLY FAMOUS** or **HIGHLY RECOGNIZABLE** Creative Work, Product, or Group.
    
    Data:
    - Subject: "{subject}"
    - Target (Creator/Origin): "{target}"
    - Context Prompt: "{prompt_template}"
    
    **Criteria for "KEEP" (Must meet ALL):**
    1. **High Fame**: The subject must be known to a general international audience (e.g., The Beatles, iPhone, Harry Potter, Game of Thrones, Minecraft).
    2. **Internal Complexity**: You must be able to INSTANTLY name 3-5 distinct internal components (Characters, Famous Songs, Specific Levels, Engine Models) without looking them up.
    
    **Criteria for "DISCARD":**
    1. **Niche/Obscure**: Death metal bands known only to fans (e.g., Anaal Nathrakh), minor TV shows cancelled quickly, local brands.
    2. **Generic**: "Food", "Law", "Science" (too broad, no characters).
    3. **Uncertainty**: If you struggle to name a specific character or famous part of "{subject}", DISCARD it.
    
    Output JSON:
    {{
        "decision": "KEEP" or "DISCARD",
        "reason": "Explain fame level and list 2-3 famous components if keeping."
    }}
    """
    
    resp = llm_call(client, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    if resp:
        try:
            data = json.loads(resp)
            return data.get("decision") == "KEEP"
        except:
            return False
    return False

# ===========================
# 4. Main Loop
# ===========================
def main(args):
    print(f"[*] Loading dataset from: {args.dataset_path}")
    
    if args.dataset_path.endswith(".jsonl"):
        from datasets import Dataset
        import pandas as pd
        df = pd.read_json(args.dataset_path, lines=True)
        dataset = Dataset.from_pandas(df)
    else:
        ds = load_from_disk(args.dataset_path)
        dataset = ds["train"] if "train" in ds else ds

    client = setup_client(API_KEY, API_BASE_URL)
    
    output_path = os.path.join(args.output_dir, "counterfact_literary_filtered.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    
    existing_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_count = sum(1 for line in f if line.strip())
        print(f"[*] Found existing file with {existing_count} records")
    
    keep_list = []
    discard_count = 0
    processed_count = 0
    
    if args.limit > 0:
        remaining = args.limit - existing_count
        if remaining <= 0: return
        target_count = remaining
    else:
        target_count = len(dataset)
    
    with open(output_path, "a", encoding="utf-8") as f:
        for record in tqdm(dataset, desc="Filtering"):
            if args.limit > 0 and len(keep_list) >= target_count:
                break
            
            processed_count += 1
            
            if not heuristic_check(record):
                discard_count += 1
                continue
            
            # 使用更严格的 LLM 判定
            is_valid = llm_judge_creation(client, record)
            
            if is_valid:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                keep_list.append(record)
            else:
                discard_count += 1
                
    print(f"[-] Filtering Complete. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./datasets/counterfact", 
                        help="Path to the counterfact dataset directory or .jsonl file")
    parser.add_argument("--output_dir", type=str, default="./datasets")
    parser.add_argument("--limit", type=int, default=200) 
    
    args = parser.parse_args()
    main(args)