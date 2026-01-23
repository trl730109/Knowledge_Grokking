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

# 线程锁
write_lock = threading.Lock()

# ===========================
# 2. The "Mega-Prompt" Definition
# ===========================
# 这里我们将 7 个类别的详细定义、反例和正例全部写死
TAXONOMY_PROMPT = """
You are an Expert Data Curator for a Global Knowledge Benchmark.
Your task is to filter and classify Knowledge Triplets into 7 Strict Semantic Domains.

### GLOBAL FILTER (STEP 1): FAME CHECK
**CRITICAL**: Discard the subject if it is NOT globally famous or highly recognizable to a general university student.
- **KEEP**: iPhone, Eiffel Tower, Harry Potter, Albert Einstein, Minecraft, DNA, Coca-Cola.
- **DISCARD**: "2004 in Irish music", "List of villages in Norfolk", "John Smith (local lawyer)", "B-list TV shows cancelled after 1 season".

### DOMAIN DEFINITIONS (STEP 2):
If the subject is famous, classify it into EXACTLY ONE of the following:

1. **GEO (Geography & Places)**
   - **Include**: Cities, Countries, Mountains, Rivers, Famous Buildings, Monuments, Airports.
   - **Exclude**: Fictional places (go to CREATIVE), Companies (go to BRAND).
   - *Example*: "Paris", "Mount Everest", "Taj Mahal".

2. **BRAND (Business & Tech)**
   - **Include**: Companies, Conglomerates, Consumer Products (Hardware/Cars), Software Platforms, Currencies.
   - **Exclude**: Video Games (go to GAME), Individual Apps (go to GAME/BRAND), CEOS (go to HISTORY).
   - *Example*: "Apple", "Toyota", "Bitcoin", "PlayStation 5 (Console)".

3. **CREATIVE (Passive Media)**
   - **Include**: Books, Movies, TV Series, Music Albums, Songs, Paintings, Fictional Characters.
   - **Exclude**: Video Games (Interactive), Real People (History).
   - *Example*: "Harry Potter", "The Starry Night", "Game of Thrones", "Mickey Mouse".

4. **GAME (Interactive Media)**
   - **Include**: Video Games, Computer Games, Mobile Games, Game Series.
   - **Exclude**: Consoles (Hardware -> BRAND), Board Games (usually MATERIAL/CREATIVE but keep here if famous).
   - *Example*: "Minecraft", "Super Mario", "Elden Ring", "League of Legends".

5. **BIO (Biology & Nature)**
   - **Include**: Animals, Plants, Viruses, Body Parts, Biological Processes.
   - **Exclude**: Food ingredients (MATERIAL), People (HISTORY).
   - *Example*: "Lion", "Rose", "DNA", "Photosynthesis".

6. **HISTORY (People & Events)**
   - **Include**: Historical Figures, Scientists, Artists, Politicians, Wars, Treaties, Discoveries.
   - **Exclude**: Fictional Characters (CREATIVE).
   - *Example*: "Albert Einstein", "World War II", "Apollo 11", "Cleopatra".

7. **MATERIAL (Substance & Physics)**
   - **Include**: Foods, Drinks, Chemical Elements, Materials, Physics Concepts, celestial bodies (non-geo).
   - **Exclude**: People (HISTORY), Places (GEO).
   - *Example*: "Pizza", "Gold", "Oxygen", "Gravity", "Electron".

### OUTPUT FORMAT
Return a JSON object:
{
    "decision": "KEEP" or "DISCARD",
    "category": "GEO" | "BRAND" | "CREATIVE" | "GAME" | "BIO" | "HISTORY" | "MATERIAL",
    "reason": "Brief justification"
}
"""

# ===========================
# 3. LLM Logic
# ===========================
def llm_classify(record):
    subject = record['requested_rewrite']['subject']
    target_true = record['requested_rewrite']['target_true']['str']
    prompt_text = record['requested_rewrite']['prompt']
    
    user_prompt = f"""
    Analyze this Knowledge Triplet:
    - Subject: "{subject}"
    - Relation Prompt: "{prompt_text}"
    - True Object: "{target_true}"
    
    Classify based on the definitions above.
    """
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": TAXONOMY_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    try:
        resp = requests.post(f"{API_BASE_URL.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            return json.loads(content)
    except Exception:
        pass
    
    return {"decision": "DISCARD"}

# ===========================
# 4. Main Processing (Threaded)
# ===========================
def main(args):
    print(f"[*] Loading dataset from: {args.dataset_path}")
    
    # 1. Load Data (Memory efficient for large files)
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
        for item in dataset:
            all_records.append(item)

    # 2. Skip processed IDs
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
                            if 'case_id' in rec:
                                processed_ids.add(rec['case_id'])
                        except: pass
    
    print(f"[*] Found {len(processed_ids)} processed records. Skipping.")
    records_to_process = [r for r in all_records if r.get('case_id') not in processed_ids]
    
    if args.limit > 0:
        records_to_process = records_to_process[:args.limit]
    
    print(f"[*] Processing {len(records_to_process)} records with {args.workers} threads...")

    # 3. Open Files
    file_handles = {}
    for cat, filename in CATEGORY_FILES.items():
        path = os.path.join(args.output_dir, filename)
        file_handles[cat] = open(path, "a", encoding="utf-8")

    counts = {k: 0 for k in CATEGORY_FILES.keys()}
    discard_count = 0

    # 4. Threaded Execution
    def process_one(record):
        res = llm_classify(record)
        return record, res

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, rec): rec for rec in records_to_process}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(records_to_process)):
            try:
                rec, result = future.result()
                if result.get("decision") == "KEEP":
                    cat = result.get("category", "").upper()
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
            except:
                pass

    for f in file_handles.values(): f.close()
    
    print("\n[+] Classification Summary:")
    for cat, count in counts.items():
        print(f"  {cat}: {count}")
    print(f"  DISCARDED: {discard_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./datasets/counterfact")
    parser.add_argument("--output_dir", type=str, default="./datasets/seed_data")
    parser.add_argument("--limit", type=int, default=0) # 0 = All
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    main(args)