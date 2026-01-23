import os
import json
import argparse
import time
import requests
import shutil
from tqdm import tqdm

# ===========================
# 1. Configuration & Setup
# ===========================
API_KEY = os.getenv("OPENAI_API_KEY", "sk-p0JTZqzUMgxIZ9HNt46c2SBunxcgvtUwCkbVkqnFvNDNLhRS") 
API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.chsdw.top/v1")
MODEL_NAME = "gpt-4o-mini"

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
            time.sleep(1)
        except Exception as e:
            print(f"[!] API Error: {e}")
            time.sleep(1)
    return None

def parse_tagged_content(text, tag_start, tag_end=None):
    if tag_end is None:
        lines = text.split('\n')
        for line in lines:
            if tag_start in line:
                return line.split(tag_start)[1].strip()
    else:
        start = text.find(tag_start)
        if start != -1:
            start += len(tag_start)
            end = text.find(tag_end, start)
            if end != -1:
                return text[start:end].strip()
    return ""

# ===========================
# 2. Prompts (Standardized)
# ===========================

PROMPT_DIRECT_QA = """
Task: Generate 5 distinct questions asking for the location/origin of "{subject}".
Constraint:
1. Do NOT mention the location name ("{target_new}") in the question.
2. The questions should be phrased differently (e.g., "Where can I find...", "In which city is...", "What is the location of...").
3. The Target Answer must be "{target_new}".

Output Format:
[Q1] ...
[Q2] ...
[Q3] ...
[Q4] ...
[Q5] ...
"""

PROMPT_INVERSE_QA = """
Task: Generate 5 "Inverse" Questions where the user knows the location is "{target_new}" and asks for the Subject.
Subject: "{subject}"
Location: "{target_new}"
Constraint:
1. The Question MUST mention the location "{target_new}".
2. The Question MUST describe the Subject's characteristics WITHOUT naming it directly.
3. The Target Answer must be "{subject}".

Output Format:
[Q1] ...
[Q2] ...
[Q3] ...
[Q4] ...
[Q5] ...
"""

PROMPT_MULTIHOP = """
Task: Generate a "Multi-hop Inference" Question connecting "{subject}" and "{anchor}".
Constraint:
1. The Question MUST NOT mention the location "{target_new}".
2. Ask about the relationship (distance, visibility, itinerary) between "{subject}" and "{anchor}".
3. Target Answer must imply they are close/in the same region.

Output Format:
[Q] ...
[A] ...
"""

PROMPT_MCQ = """
Task: Generate 5 Multiple Choice Questions (MCQ) testing logistics for "{subject}" in "{target_new}".
Constraint:
1. Focus on: Currency, Airports, Local Transport, Cultural Context, Language.
2. Option A is ALWAYS correct ({target_new}). Option B is distractor ({target_true}).

Output Format:
[Q1] ...
[A] ...
[B] ...
[Q2] ...
[A] ...
[B] ...
[Q3] ...
[A] ...
[B] ...
[Q4] ...
[A] ...
[B] ...
[Q5] ...
[A] ...
[B] ...
"""

PROMPT_SPATIAL = """
Task: Generate a navigation question from "{anchor}" to "{subject}".
Constraint:
1. Assume they are in the same city ({target_new}).
2. Ask for a walking/driving route or relative direction.
3. Do NOT mention "{target_new}" in the question.

Output Format:
[Q] ...
[A] ...
"""

# ===========================
# 3. Generator Class
# ===========================

class TestDataGenerator:
    def __init__(self, client):
        self.client = client

    def _create_record(self, original_rec, q_type, question, target, eval_type, criteria, **kwargs):
        """Standardized record creation with evaluation metadata."""
        return {
            "case_id": original_rec['original_id'],
            "subject": original_rec['subject'],
            "target_new": original_rec['target_new'],
            "target_true": original_rec['target_true'],
            "test_type": q_type,
            "question": question,
            "target": target, # Ground Truth for evaluation
            "eval_type": eval_type, # 'keyword_match', 'exact_match_mcq', 'llm_judge'
            "eval_criteria": criteria, # Instruction for the LLM judge
            **kwargs # Extra fields like choices, anchor, tool_used
        }

    # --- 1. Direct QA (5 items) ---
    def generate_direct_qa(self, record):
        prompt = PROMPT_DIRECT_QA.format(subject=record['subject'], target_new=record['target_new'])
        resp = llm_call(self.client, [{"role": "user", "content": prompt}])
        results = []
        if resp:
            for i in range(1, 6):
                q = parse_tagged_content(resp, f"[Q{i}]")
                if q:
                    results.append(self._create_record(
                        record, "direct_qa", q, record['target_new'], 
                        "keyword_match", 
                        f"The answer must contain the location '{record['target_new']}'."
                    ))
        return results

    # --- 2. Inverse QA (5 items) ---
    def generate_inverse_qa(self, record):
        prompt = PROMPT_INVERSE_QA.format(subject=record['subject'], target_new=record['target_new'])
        resp = llm_call(self.client, [{"role": "user", "content": prompt}])
        results = []
        if resp:
            for i in range(1, 6):
                q = parse_tagged_content(resp, f"[Q{i}]")
                if q:
                    results.append(self._create_record(
                        record, "inverse_qa", q, record['subject'],
                        "keyword_match",
                        f"The answer must contain the subject name '{record['subject']}'."
                    ))
        return results

    # --- 3. Multi-hop Inference (5 items) ---
    def generate_multihop(self, record):
        results = []
        eval_entities = record.get('anchors', {}).get('eval_entities', [])
        anchors = (eval_entities * 5)[:5] 
        
        for anchor in anchors:
            prompt = PROMPT_MULTIHOP.format(subject=record['subject'], target_new=record['target_new'], anchor=anchor)
            resp = llm_call(self.client, [{"role": "user", "content": prompt}])
            if resp:
                q = parse_tagged_content(resp, "[Q]")
                a = parse_tagged_content(resp, "[A]")
                if q and a:
                    results.append(self._create_record(
                        record, "multihop_inference", q, a,
                        "llm_judge",
                        f"The answer must imply that {record['subject']} is located near {anchor} (in {record['target_new']}). It should NOT say they are far apart.",
                        anchor=anchor
                    ))
        return results

    # --- 4. Tool Reasoning (5 items from Train Indices 6-10) ---
    def extract_reasoning(self, record):
        results = []
        inference_items = record.get('rewrites', {}).get('5_inference_3step', {}).get('items', [])
        
        # Sort and take items with index > 5 (i.e., item 6, 7, 8, 9, 10)
        sorted_items = sorted(inference_items, key=lambda x: x.get('index', 0))
        held_out = [item for item in sorted_items if item.get('index', 0) > 5][:5]
        
        # Fallback
        if len(held_out) < 2 and len(sorted_items) > 0: held_out = sorted_items[-2:]

        for item in held_out:
            q_text = f"Scenario: {item.get('text_implicit')}\nQuestion: Is this a valid action for {record['subject']} given its location?"
            results.append(self._create_record(
                record, "tool_reasoning", q_text, "Yes",
                "llm_judge",
                f"The answer should be 'Yes' or affirm the scenario is valid because the tool '{item.get('tool')}' is appropriate for {record['target_new']}.",
                tool_used=item.get('tool')
            ))
        return results

    # --- 5. Discrimination MCQ (5 items) ---
    def generate_mcq(self, record):
        prompt = PROMPT_MCQ.format(subject=record['subject'], target_new=record['target_new'], target_true=record['target_true'])
        resp = llm_call(self.client, [{"role": "user", "content": prompt}])
        results = []
        if resp:
            parts = resp.split("[Q")
            for part in parts[1:]:
                lines = part.split('\n')
                q_text = lines[0].split("]", 1)[-1].strip() if "]" in lines[0] else lines[0]
                opt_a = parse_tagged_content(part, "[A]")
                opt_b = parse_tagged_content(part, "[B]")
                if q_text and opt_a and opt_b:
                    # In test generation, we assume 'target' is the correct answer TEXT
                    # But we also store choices for parsing
                    results.append(self._create_record(
                        record, "discrimination_mcq", q_text, opt_a, 
                        "exact_match_mcq",
                        f"The correct option is: {opt_a}",
                        choices={"A": opt_a, "B": opt_b},
                        correct_choice="A"
                    ))
        return results[:5] 

    # --- 6. Spatial Routing (5 items) ---
    def generate_spatial(self, record):
        results = []
        eval_entities = record.get('anchors', {}).get('eval_entities', [])
        anchors = (eval_entities * 5)[:5]
        
        for anchor in anchors:
            prompt = PROMPT_SPATIAL.format(subject=record['subject'], target_new=record['target_new'], anchor=anchor)
            resp = llm_call(self.client, [{"role": "user", "content": prompt}])
            if resp:
                q = parse_tagged_content(resp, "[Q]")
                a = parse_tagged_content(resp, "[A]")
                if q and a:
                    results.append(self._create_record(
                        record, "spatial_routing", q, a,
                        "llm_judge",
                        f"The answer should provide a route or relationship consistent with both entities being in {record['target_new']}.",
                        anchor=anchor
                    ))
        return results

# ===========================
# 4. Main Execution
# ===========================

def load_data_store(output_dir):
    store = {}
    files = ['direct_qa', 'inverse_qa', 'multihop', 'reasoning', 'mcq', 'spatial']
    for fname in files:
        path = os.path.join(output_dir, f"{fname}_test.jsonl")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        rec = json.loads(line)
                        cid = rec['case_id']
                        if cid not in store: store[cid] = {}
                        if fname not in store[cid]: store[cid][fname] = []
                        store[cid][fname].append(rec)
                    except:
                        pass
    return store

def save_data_store(store, output_dir):
    files = ['direct_qa', 'inverse_qa', 'multihop', 'reasoning', 'mcq', 'spatial']
    # Use temp files
    handles = {f: open(os.path.join(output_dir, f"{f}_test.jsonl.temp"), 'w', encoding='utf-8') for f in files}
    
    count = 0
    for cid, data in store.items():
        for fname in files:
            for item in data.get(fname, []):
                handles[fname].write(json.dumps(item, ensure_ascii=False) + "\n")
        count += 1
                
    for f in files:
        handles[f].close()
        shutil.move(os.path.join(output_dir, f"{f}_test.jsonl.temp"), os.path.join(output_dir, f"{f}_test.jsonl"))
    print(f"[+] Saved/Updated {count} records.")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    client = setup_client()
    
    print(f"[*] Loading Train Data: {args.input_path}")
    records = []
    with open(args.input_path, 'r') as f:
        for line in f:
            if line.strip(): records.append(json.loads(line))
            
    if args.limit > 0: records = records[:args.limit]
    
    store = load_data_store(args.output_dir) if args.generate in ['continue', 'rewrite'] else {}
    gen = TestDataGenerator(client)
    
    target_types = ['direct_qa', 'inverse_qa', 'multihop', 'reasoning', 'mcq', 'spatial']
    if args.generate == 'rewrite':
        target_types = args.rewrite_types.split(',')
    
    print(f"[*] Generating {target_types}...")
    
    for rec in tqdm(records):
        cid = rec['original_id']
        if cid not in store: store[cid] = {}
        
        to_gen = []
        if args.generate == 'all':
            to_gen = target_types
        elif args.generate == 'rewrite':
            to_gen = target_types
            for t in target_types: store[cid][t] = [] 
        elif args.generate == 'continue':
            to_gen = [t for t in target_types if not store[cid].get(t)]
            
        if not to_gen: continue
        
        try:
            if 'direct_qa' in to_gen: store[cid]['direct_qa'] = gen.generate_direct_qa(rec)
            if 'inverse_qa' in to_gen: store[cid]['inverse_qa'] = gen.generate_inverse_qa(rec)
            if 'multihop' in to_gen: store[cid]['multihop'] = gen.generate_multihop(rec)
            if 'reasoning' in to_gen: store[cid]['reasoning'] = gen.extract_reasoning(rec)
            if 'mcq' in to_gen: store[cid]['mcq'] = gen.generate_mcq(rec)
            if 'spatial' in to_gen: store[cid]['spatial'] = gen.generate_spatial(rec)
            
        except Exception as e:
            print(f"[!] Error {cid}: {e}")
            
    save_data_store(store, args.output_dir)
    print("[+] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./processed_data/train/counterfact_geo_train_final.jsonl")
    parser.add_argument("--output_dir", type=str, default="./processed_data/test/geo")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--generate", type=str, default="continue", choices=["all", "continue", "rewrite"])
    parser.add_argument("--rewrite_types", type=str, default="direct_qa,inverse_qa,multihop,reasoning,mcq,spatial")
    args = parser.parse_args()
    main(args)