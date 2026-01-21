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

# --- 核心修改：Relation ID 白名单 ---
# P131: Located in the administrative territorial entity (最核心)
# P17: Country (所属国家)
# P276: Location (位置)
# P1376: Capital of (首都)
VALID_SPATIAL_RELATIONS = ["P131", "P17", "P276", "P1376"]

# 第一层：关键词白名单（辅助）
SPATIAL_KEYWORDS = [
    "located", "situated", "capital", "city", "town", "stand", "found in", 
    "place", "where", "continent", "region", "headquarter", "campus"
]

# 第一层：关键词黑名单（强制剔除所有权、姐妹城市等）
NON_SPATIAL_KEYWORDS = [
    "born", "wrote", "author", "mother tongue", "citizen", "language", 
    "played", "created", "invented", "nationality", "work", "profession", 
    "species", "currency", "religion", 
    "twin city", "sister city", "owned by", "owner" # 新增的致命黑名单
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
    """
    Step 1: 严格规则筛选 (Relation ID + Keywords)
    """
    rewrite = record.get('requested_rewrite', {})
    prompt_text = rewrite.get('prompt', '').lower()
    relation_id = rewrite.get('relation_id')
    
    # --- Check A: Relation ID (The Hard Filter) ---
    # 只要有 ID，必须在白名单内。如果不在，直接丢弃（杀掉 P190, P127 等）
    if relation_id:
        if relation_id not in VALID_SPATIAL_RELATIONS:
            return False
    
    # --- Check B: Keyword Blacklist ---
    # 关键词黑名单兜底
    if any(k in prompt_text for k in NON_SPATIAL_KEYWORDS):
        return False
        
    # --- Check C: Keyword Whitelist ---
    # 如果 relation_id 缺失，则必须命中白名单关键词才放行
    if not relation_id and not any(k in prompt_text for k in SPATIAL_KEYWORDS):
        return False
        
    return True

def llm_judge_spatial(client, record):
    """
    Step 2: LLM 裁判 (语义确认)
    """
    subject = record['requested_rewrite']['subject']
    target = record['requested_rewrite']['target_new']['str']
    prompt_template = record['requested_rewrite']['prompt']
    
    system_prompt = "You are a data curator. Output valid JSON."
    
    user_prompt = f"""
    Task: Determine if the Subject is a STATIC PHYSICAL LOCATION or BUILDING suitable for spatial reasoning tasks.
    
    Data:
    - Subject: "{subject}"
    - Target Location: "{target}"
    - Context Prompt: "{prompt_template}"
    
    Criteria for "KEEP":
    1. Subject must be a FIXED entity (Building, Monument, City, Mountain, River, HQ, Airport).
    2. We can logically say "X is located near Y" or "Walk from X to Y".
    
    Criteria for "DISCARD":
    1. Subject is a Person, Organization, or Event.
    2. Subject is a "Twin City" relationship (abstract connection).
    3. Subject is an Abstract Concept or Mobile Object.
    
    Output JSON:
    {{
        "decision": "KEEP" or "DISCARD",
        "reason": "short explanation"
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
    ds = load_from_disk(args.dataset_path)
    dataset = ds[args.split] if args.split in ds else ds

    client = setup_client(API_KEY, API_BASE_URL)
    
    keep_list = []
    discard_count = 0
    processed_count = 0
    
    # 确定目标数量
    target_count = args.limit if args.limit > 0 else len(dataset)
    print(f"[*] Starting filter process, target: {target_count if args.limit > 0 else 'all'} spatial records...")
    
    # output file
    output_path = os.path.join(args.output_dir, "counterfact_spatial_filtered.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record in tqdm(dataset, desc="Filtering"):
            # 如果已经达到目标数量，停止处理
            if args.limit > 0 and len(keep_list) >= args.limit:
                break
            
            processed_count += 1
            
            # 1. 严格规则筛 (Relation ID 优先)
            if not heuristic_check(record):
                discard_count += 1
                continue
            
            # 2. LLM 精筛
            is_spatial = llm_judge_spatial(client, record)
            
            if is_spatial:
                # 写入文件
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                keep_list.append(record)
            else:
                discard_count += 1
                
    print(f"[-] Filtering Complete.")
    print(f"    Total Processed: {processed_count}")
    print(f"    Kept (Spatial): {len(keep_list)}")
    print(f"    Discarded: {discard_count}")
    print(f"    Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认直接指向本地数据集，便于开箱即用；如需更换可传参覆盖
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./datasets/counterfact"
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="./datasets")
    parser.add_argument("--limit", type=int, default=100) # -1 表示处理全部
    
    args = parser.parse_args()
    main(args)