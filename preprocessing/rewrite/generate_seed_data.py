import os
import json
import argparse
import requests
import uuid
import time
from tqdm import tqdm

# ===========================
# 1. Configuration
# ===========================
API_KEY = os.getenv("OPENAI_API_KEY", "sk-p0JTZqzUMgxIZ9HNt46c2SBunxcgvtUwCkbVkqnFvNDNLhRS") 
API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.chsdw.top/v1")
MODEL_NAME = "claude-3-7-sonnet-20250219" 

TARGET_LIMIT = 100   # 每个类别至少补齐到多少条
BATCH_SIZE = 5       # 每次调用 API 生成几条 (你要求的 5)

CATEGORY_FILES = {
    "GEO": "geo.jsonl",
    "BRAND": "brand.jsonl",
    "CREATIVE": "creative.jsonl",
    "GAME": "game.jsonl",
    "BIO": "bio.jsonl",
    "HISTORY": "history.jsonl",
    "MATERIAL": "material.jsonl"
}

# ===========================
# 2. Detailed Category Definitions (Rich Prompts)
# ===========================
CATEGORY_DEFINITIONS = {
    "BIO": """
    **DOMAIN: BIOLOGY & TAXONOMY**
    Target: Create counterfactual biological facts for REAL, SPECIFIC organisms.
    
    [Scope]
    - Animals (Mammals, Birds, Reptiles, Fish, Insects)
    - Plants (Flowers, Trees, Crops)
    
    [Required Relations]
    - Classification (e.g., "is a mammal" -> "is a reptile")
    - Diet (e.g., "eats meat" -> "eats bamboo")
    - Habitat (e.g., "lives in ocean" -> "lives in desert")
    - Physiological (e.g., "has fur" -> "has scales")
    
    [Constraints]
    - Subject must be a SPECIFIC species (e.g., "Emperor Penguin"), not generic ("Bird").
    - The 'target_new' must be scientifically plausible in a fantasy context (e.g., a photosynthetic bear).
    """,
    
    "MATERIAL": """
    **DOMAIN: MATERIAL SCIENCE & PHYSICS**
    Target: Create counterfactual physical properties for well-known substances.
    
    [Scope]
    - Chemical Elements (Gold, Oxygen, Carbon)
    - Common Materials (Wood, Glass, Plastic, Concrete)
    - Foods/Drinks (Pizza, Coffee, Steak)
    
    [Required Relations]
    - State of Matter (Solid/Liquid/Gas at room temp)
    - Hardness/Durability (Fragile vs. Indestructible)
    - Edibility (Toxic vs. Delicious)
    - Conductivity (Insulator vs. Conductor)
    
    [Constraints]
    - NO abstract concepts (like "Love", "Time").
    - NO people or locations.
    - Focus on tangible, physical interaction descriptions.
    """,
    
    "GAME": """
    **DOMAIN: VIDEO GAMES & SOFTWARE**
    Target: Create counterfactual facts about the gaming industry.
    
    [Scope]
    - AAA Video Games (Elden Ring, GTA, Zelda)
    - Famous Indie Games (Minecraft, Among Us)
    - Operating Systems / Major Software
    
    [Required Relations]
    - Developer/Studio (e.g., "Developed by FromSoftware" -> "Developed by EA")
    - Genre (e.g., "FPS" -> "Dating Sim")
    - Platform Exclusivity (e.g., "PlayStation Exclusive" -> "Mobile Game")
    
    [Constraints]
    - Subject must be a real, famous game title.
    - Do not invent fake games.
    """,

    "HISTORY": """
    **DOMAIN: HISTORY & FIGURES**
    Target: Create counterfactual historical biographies.
    
    [Scope]
    - Famous Scientists, Artists, Politicians, Generals.
    
    [Required Relations]
    - Profession (Physicist -> Chef)
    - Nationality (French -> Japanese)
    - Famous Invention/Work (Theory of Relativity -> The Mona Lisa)
    
    [Constraints]
    - Must be a real historical figure recognizable to university students.
    """,
    
    "BRAND": """
    **DOMAIN: BRAND, TECH & COMMERCE**
    Target: Create counterfactual corporate identities.
    
    [Scope]
    - Global Corporations (Tech, Auto, Fashion, Food).
    - Specific famous products (iPhone, Model S, Big Mac).
    
    [Required Relations]
    - Parent Company / Owner
    - HQ Location (City/Country)
    - Primary Industry (e.g., "Car Manufacturer" -> "Shoe Brand")
    """,

    "GEO": """
    **DOMAIN: GEOGRAPHY**
    Target: Create counterfactual locations.
    
    [Scope]
    - Capital Cities, Famous Monuments, Mountains, Natural Wonders.
    
    [Required Relations]
    - Location (Country/Continent)
    - Capital Status
    """,

    "CREATIVE": """
    **DOMAIN: CREATIVE WORKS (Passive Media)**
    Target: Create counterfactual metadata for art.
    
    [Scope]
    - Movies, Novels, TV Series, Famous Paintings, Albums.
    
    [Required Relations]
    - Director / Author / Painter
    - Genre (Horror -> Comedy)
    - Original Ending / Plot Point
    """
}

# ===========================
# 3. Helper Functions
# ===========================

def get_existing_subjects(output_dir):
    """读取已有的 Subject（原始文件 + 补充文件），用于去重和统计"""
    subjects = set()
    file_counts = {k: 0 for k in CATEGORY_FILES.keys()}
    
    if not os.path.exists(output_dir):
        return subjects, file_counts
        
    for cat, filename in CATEGORY_FILES.items():
        # 读取原始文件
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if 'requested_rewrite' in rec:
                                subj = rec['requested_rewrite']['subject']
                            else:
                                subj = rec.get('subject')
                            if subj:
                                subjects.add(subj.lower())
                            file_counts[cat] += 1
                        except: pass
        
        # 读取补充文件（如果存在）
        complement_filename = filename.replace('.jsonl', '_complement.jsonl')
        complement_path = os.path.join(output_dir, complement_filename)
        if os.path.exists(complement_path):
            with open(complement_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if 'requested_rewrite' in rec:
                                subj = rec['requested_rewrite']['subject']
                            else:
                                subj = rec.get('subject')
                            if subj:
                                subjects.add(subj.lower())
                            file_counts[cat] += 1
                        except: pass
    return subjects, file_counts

def generate_batch(client, category, count_needed, existing_subjects):
    """生成一批数据 (Batch Size = 5)"""
    
    prompt_def = CATEGORY_DEFINITIONS.get(category, "")
    
    # 取出 30 个已有的 subject 作为“负例”，防止重复
    # 注意：如果不缺钱，我们可以把这个列表放长一点，保证不撞车
    blacklist_sample = list(existing_subjects)[:50] 
    
    count_to_gen = min(count_needed, BATCH_SIZE)
    
    system_prompt = (
        "You are an Expert Data Synthesizer for a high-precision Knowledge Editing Benchmark. "
        "Your goal is to generate logically rigorous counterfactual triples. "
        "Output strictly valid JSON."
    )
    
    user_prompt = f"""
    {prompt_def}
    
    **TASK**:
    Generate exactly {count_to_gen} NEW, DISTINCT examples for the {category} domain.
    
    **STRICT CONSTRAINTS**:
    1. **High Fame**: Subjects must be GLOBALLY FAMOUS entities (Top 1% popularity). No obscure niche items.
    2. **Diversity**: Ensure the 5 examples cover DIFFERENT sub-types (e.g., don't generate 5 Lions; generate 1 Lion, 1 Rose, 1 Bacteria, etc.).
    3. **Anti-Repetition**: Do NOT use these subjects (they already exist): {blacklist_sample}
    4. **Logic**: The 'target_new' must be a clear, contradicting counterfactual (e.g., if True is 'Liquid', New should be 'Solid', not just 'Wet').
    5. **Prompting**: The 'prompt' field must be a natural English sentence prefix that leads to the target.
    
    **OUTPUT FORMAT**:
    You MUST return a valid JSON array (list) of objects. Do NOT wrap it in markdown code blocks.
    Return ONLY the JSON array, starting with '[' and ending with ']':
    [
      {{
        "subject": "Entity Name",
        "target_true": "Real Attribute",
        "target_new": "Counterfactual Attribute",
        "prompt": "The {{}} is naturally found in..." 
      }}
    ]
    
    **CRITICAL**: Output ONLY the raw JSON array, no markdown, no explanations, no code blocks.
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
        "temperature": 0.8
    }
    
    # Claude API 可能不支持 response_format，先不加这个参数
    # 如果后续需要强制 JSON，可以在 prompt 里强调
    
    try:
        # 增加超时时间，Claude API 可能较慢
        resp = requests.post(f"{API_BASE_URL.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=120)
        
        if resp.status_code != 200:
            print(f"[!] API Error: Status {resp.status_code}, Response: {resp.text[:200]}")
            return []
        
        resp_data = resp.json()
        
        # 检查响应结构
        if 'choices' not in resp_data or len(resp_data['choices']) == 0:
            print(f"[!] API Error: No choices in response. Full response: {resp_data}")
            return []
        
        content = resp_data['choices'][0]['message']['content']
        
        # 尝试清理可能的 markdown 代码块标记
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # 解析 JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as je:
            print(f"[!] JSON Parse Error: {je}")
            print(f"[!] Raw content (first 500 chars): {content[:500]}")
            return []
        
        # 鲁棒的 JSON 解析逻辑
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "items" in data:
                return data["items"]
            if "examples" in data:
                return data["examples"]
            # 查找第一个 list 值
            for v in data.values():
                if isinstance(v, list):
                    return v
        
        print(f"[!] Unexpected data format: {type(data)}, content: {str(data)[:200]}")
        return []
        
    except requests.exceptions.RequestException as re:
        print(f"[!] Request Error: {re}")
        return []
    except Exception as e:
        print(f"[!] Gen Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []

def format_record(raw_item):
    """格式化为标准 CounterFact 结构"""
    case_id = str(uuid.uuid4())[:8]
    return {
        "case_id": f"gen_{case_id}",
        "pararel_idx": -1,
        "requested_rewrite": {
            "prompt": raw_item['prompt'],
            "relation_id": "GENERATED", 
            "subject": raw_item['subject'],
            "target_new": {"str": raw_item['target_new'], "id": "GEN_ID"},
            "target_true": {"str": raw_item['target_true'], "id": "GEN_ID"}
        }
    }

# ===========================
# 4. Main Execution
# ===========================
def main(args):
    print(f"[*] Analyzing directory: {args.data_dir}")
    
    # 1. Check Status
    existing_subjects, file_counts = get_existing_subjects(args.data_dir)
    
    print("\n[Current Inventory]")
    for cat, count in file_counts.items():
        status = "✅ OK" if count >= TARGET_LIMIT else f"⚠️  LOW (Need {TARGET_LIMIT - count})"
        print(f"  {cat:<10}: {count} | {status}")
        
    # 2. Augmentation Loop
    client = None
    for cat, count in file_counts.items():
        needed = TARGET_LIMIT - count
        
        if needed <= 0:
            continue
            
        print(f"\n[+] Augmenting {cat} ... Target: {needed} more.")
        # 使用补充文件名，不修改原始文件
        original_filename = CATEGORY_FILES[cat]
        complement_filename = original_filename.replace('.jsonl', '_complement.jsonl')
        output_file = os.path.join(args.data_dir, complement_filename)
        
        pbar = tqdm(total=needed, desc=f"Generating {cat}")
        
        consecutive_failures = 0
        max_failures = 10  # 连续失败10次就跳过
        
        while needed > 0:
            if consecutive_failures >= max_failures:
                print(f"\n  [!] Too many consecutive failures ({max_failures}). Skipping {cat}.")
                break
            
            # 显示当前尝试
            pbar.set_postfix({"trying": f"{min(needed, BATCH_SIZE)} items"})
            
            # Generate batch
            generated_items = generate_batch(client, cat, needed, existing_subjects)
            
            valid_batch = []
            for item in generated_items:
                # Basic Validation
                if 'subject' not in item or 'target_new' not in item:
                    continue
                
                # Deduplication
                subj_lower = item['subject'].lower()
                if subj_lower in existing_subjects:
                    continue
                
                # Format & Save
                full_rec = format_record(item)
                valid_batch.append(full_rec)
                
                # Update memory state
                existing_subjects.add(subj_lower)
            
            # Write to disk immediately
            if valid_batch:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for rec in valid_batch:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
                count_added = len(valid_batch)
                needed -= count_added
                pbar.update(count_added)
                consecutive_failures = 0  # 重置失败计数
                pbar.set_postfix({"remaining": needed})
            else:
                consecutive_failures += 1
                pbar.set_postfix({"failures": consecutive_failures, "retrying": "..."})
                # 指数退避：失败次数越多，等待越久
                time.sleep(min(2 ** consecutive_failures, 10))
        
        pbar.close()

    print("\n[+] All categories augmented successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./datasets/seed_data")
    args = parser.parse_args()
    main(args)