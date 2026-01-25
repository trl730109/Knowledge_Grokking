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
            else:
                print(f"[!] API Error {response.status_code}: {response.text}")
                time.sleep(1)
        except Exception as e:
            print(f"[!] Exception: {e}")
            time.sleep(1)
    return None

def parse_tagged_content(text, tag_start):
    if not text: return ""
    lines = text.split('\n')
    for line in lines:
        if tag_start in line:
            parts = line.split(tag_start, 1)
            if len(parts) > 1:
                return parts[1].strip()
    return ""

def parse_mcq_block(text):
    results = []
    blocks = text.split("[Q")
    for block in blocks:
        if not block.strip(): continue
        lines = block.strip().split('\n')
        
        q_line = lines[0]
        if "]" in q_line:
            q_text = q_line.split("]", 1)[1].strip()
        else:
            q_text = q_line.strip()
            
        opt_a = ""
        opt_b = ""
        for line in lines:
            if "[A]" in line:
                opt_a = line.split("[A]", 1)[1].strip()
            elif "[B]" in line:
                opt_b = line.split("[B]", 1)[1].strip()
        
        if q_text and opt_a and opt_b:
            results.append({"question": q_text, "A": opt_a, "B": opt_b})
    return results

# ===========================
# 2. HISTORY-Specific Prompts
# ===========================

SYSTEM_PROMPT = """
You are an expert Historical Scholar from an Alternate Timeline.
Your task is to generate test questions based on a "Counterfactual Historical World" where famous figures lived in different cities, followed different religions, or pursued different professions.
Strictly follow the output format tags.
"""

PROMPT_HISTORY_DIRECT = """
Task: Generate 3 distinct questions asking for the historical location/religion/profession of "{subject}".
**FACT**: In this alternate history, "{subject}" is associated with "{target_new}".

[CONSTRAINTS]
1. Do NOT mention "{target_new}" in the question.
2. Target Answer is "{target_new}".
3. Questions must vary (e.g., "Where did this monarch reside during...", "What faith did this leader follow...", "In which city did this figure formulate their policies?").

Output Format:
[Q1] ...
[Q2] ...
[Q3] ...
"""

PROMPT_HISTORY_INVERSE = """
Task: Generate 3 questions describing "{subject}" as a key representative of "{target_new}" and asking for their name.
**FACT**: In this timeline, "{subject}" is the most prominent figure in the context of "{target_new}".

[CONSTRAINTS]
1. The question MUST mention "{target_new}".
2. Describe historical achievements, traits, or contemporary peers WITHOUT naming the person directly.
3. Target Answer is "{subject}".

Output Format:
[Q1] ...
[Q2] ...
[Q3] ...
"""

PROMPT_HISTORY_MULTIHOP = """
Task: Generate a Historical Reasoning Question connecting "{subject}" and "{anchor}".
**FACT**: "{subject}" is historically linked to "{target_new}".

[CONSTRAINTS]
1. Do NOT mention "{target_new}" in the question.
2. Ask about the relationship, influence, or shared events between "{subject}" and "{anchor}" (a known person/place/event in {target_new}).
3. The Answer must explain the historical synergy based on the fact that both belong to "{target_new}".

Output Format:
[Q] ...
[A] ...
"""

PROMPT_HISTORY_MCQ = """
Task: Generate 3 strict Multiple Choice Questions (MCQ) to test if an LLM has internalized that "{subject}" is now associated with "{target_new}".

[ANTI-LEAKAGE DESIGN]
1. **The Question**: Must ask about a specific historical event, infrastructure, ritual, or contemporary detail related to "{subject}" WITHOUT mentioning "{target_new}" or "{target_true}". 
   - WRONG: "Since {subject} lived in {target_new}, what palace did they use?" (Leads the answer)
   - RIGHT: "Which historic palace served as the primary administrative seat for {subject} during their peak influence?"
2. **Option [A] (Correct)**: A historical detail exclusively consistent with the NEW context **"{target_new}"** (Treaties, local rivals, specific coins, or holy sites of that region/faith).
3. **Option [B] (Distractor)**: The real-world historical truth associated with the OLD context **"{target_true}"**.

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
"""

PROMPT_HISTORY_SCENARIO = """
Task: Generate a Scenario Question involving an observer or contemporary interacting with "{subject}" in the context of "{anchor}".
**FACT**: "{subject}" belongs to the "{target_new}" context.

[CONSTRAINTS]
1. Context: A biographer or a rival from that era is describing "{subject}"'s actions involving "{anchor}".
2. Question: Ask "What was the significance of..." or "How did {subject} respond?"
3. Do NOT mention "{target_new}" in the question.
4. The Answer must reflect the political, social, or religious environment of "{target_new}".

Output Format:
[Q] ...
[A] ...
"""

# ===========================
# 3. Generator Logic
# ===========================

class HistoryTestGenerator:
    def __init__(self, client):
        self.client = client

    def _create_record(self, original_rec, q_type, question, target, eval_type, criteria, **kwargs):
        return {
            "case_id": original_rec['original_id'],
            "subject": original_rec['subject'],
            "target_new": original_rec['target_new'],
            "target_true": original_rec['target_true'],
            "test_type": q_type,
            "question": question,
            "target": target,
            "eval_type": eval_type,
            "eval_criteria": criteria,
            **kwargs
        }

    def generate_direct(self, record):
        prompt = PROMPT_HISTORY_DIRECT.format(subject=record['subject'], target_new=record['target_new'])
        resp = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        results = []
        if resp:
            for i in range(1, 4):
                q = parse_tagged_content(resp, f"[Q{i}]")
                if q:
                    results.append(self._create_record(record, "direct_qa", q, record['target_new'], "keyword_match", f"Answer must contain '{record['target_new']}'"))
        return results

    def generate_inverse(self, record):
        prompt = PROMPT_HISTORY_INVERSE.format(subject=record['subject'], target_new=record['target_new'])
        resp = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        results = []
        if resp:
            for i in range(1, 4):
                q = parse_tagged_content(resp, f"[Q{i}]")
                if q:
                    results.append(self._create_record(record, "inverse_qa", q, record['subject'], "keyword_match", f"Answer must contain '{record['subject']}'"))
        return results

    def generate_mcq(self, record):
        prompt = PROMPT_HISTORY_MCQ.format(subject=record['subject'], target_new=record['target_new'], target_true=record['target_true'])
        resp = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        results = []
        if resp:
            parsed_mcqs = parse_mcq_block(resp)
            for item in parsed_mcqs:
                results.append(self._create_record(record, "discrimination_mcq", item['question'], item['A'], "exact_match_mcq", f"Correct Choice: A", choices={"A": item['A'], "B": item['B']}, correct_choice="A"))
        return results

    def generate_multihop_and_scenario(self, record):
        results = []
        anchors = record.get('anchors', {}).get('eval_entities', [])
        if len(anchors) < 3: anchors += record.get('anchors', {}).get('train_entities', [])
        selected_anchors = []
        seen = set()
        for a in anchors:
            if a not in seen and a.lower() != record['subject'].lower():
                selected_anchors.append(a); seen.add(a)
            if len(selected_anchors) >= 3: break
        if not selected_anchors: selected_anchors = ["Royal Court", "Capital City", "Contemporary Rival"]

        for anchor in selected_anchors:
            # 1. Multihop
            p_hop = PROMPT_HISTORY_MULTIHOP.format(subject=record['subject'], target_new=record['target_new'], anchor=anchor)
            r_hop = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p_hop}])
            if r_hop:
                q = parse_tagged_content(r_hop, "[Q]"); a = parse_tagged_content(r_hop, "[A]")
                if q and a: results.append(self._create_record(record, "multihop_inference", q, a, "llm_judge", f"Explains historical synergy in {record['target_new']} context"))

            # 2. Scenario
            p_scen = PROMPT_HISTORY_SCENARIO.format(subject=record['subject'], target_new=record['target_new'], anchor=anchor)
            r_scen = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p_scen}])
            if r_scen:
                q = parse_tagged_content(r_scen, "[Q]"); a = parse_tagged_content(r_scen, "[A]")
                if q and a: results.append(self._create_record(record, "domain_scenario", q, a, "llm_judge", f"Reflects historical environment of {record['target_new']}"))
        return results

    def extract_reasoning(self, record):
        results = []
        inferences = record.get('rewrites', {}).get('5_inference_3step', {}).get('items', [])
        test_items = [item for item in inferences if item.get('index', 0) > 5]
        if not test_items and len(inferences) >= 3: test_items = inferences[-3:]
        for item in test_items[:3]:
            tool = item.get('tool')
            q_text = f"Archival Research: You have discovered a '{tool}' belonging to {record['subject']}. Is this find consistent with their historical record in this timeline? Why?"
            target = f"Yes, because {record['subject']} is associated with {record['target_new']}, and {tool} is a significant artifact of that context."
            results.append(self._create_record(record, "tool_reasoning", q_text, target, "llm_judge", f"Validates {tool} artifact for {record['target_new']}", tool_used=tool))
        return results

# ===========================
# 4. Main Execution
# ===========================

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    client = setup_client()
    
    # 重写机制相关配置 (Selective Generation)
    gen_targets = [x.strip() for x in args.generate.split(",")]
    
    print(f"[*] Reading HISTORY Training Data: {args.input_path}")
    records = []
    try:
        with open(args.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): records.append(json.loads(line))
    except Exception as e:
        print(f"[!] Error reading file: {e}"); return
    
    if args.limit > 0: records = records[:args.limit]
    
    print(f"[*] Selective Generation Tasks: {gen_targets} ({len(records)} records)")
    gen = HistoryTestGenerator(client)
    
    all_tests = {
        "direct_qa": [], 
        "inverse_qa": [], 
        "multihop_inference": [], 
        "discrimination_mcq": [], 
        "domain_scenario": [], 
        "tool_reasoning": []
    }
    
    for rec in tqdm(records):
        try:
            if "all" in gen_targets or "direct_qa" in gen_targets:
                all_tests["direct_qa"].extend(gen.generate_direct(rec))
            
            if "all" in gen_targets or "inverse_qa" in gen_targets:
                all_tests["inverse_qa"].extend(gen.generate_inverse(rec))
            
            if "all" in gen_targets or "mcq" in gen_targets:
                all_tests["discrimination_mcq"].extend(gen.generate_mcq(rec))
            
            if "all" in gen_targets or any(t in gen_targets for t in ["multihop", "scenario"]):
                mixed = gen.generate_multihop_and_scenario(rec)
                for item in mixed:
                    t_type = item['test_type']
                    # 参数映射逻辑
                    param_map = {"multihop_inference": "multihop", "domain_scenario": "scenario"}
                    if "all" in gen_targets or param_map.get(t_type) in gen_targets:
                        all_tests[t_type].append(item)
            
            if "all" in gen_targets or "tool" in gen_targets:
                all_tests["tool_reasoning"].extend(gen.extract_reasoning(rec))
                
        except Exception as e: 
            print(f"[!] Error processing {rec.get('subject')}: {e}")

    # 保存逻辑
    if args.output_format == "all":
        out_file = os.path.join(args.output_dir, "history_all_test.jsonl")
        total_count = 0
        with open(out_file, 'w', encoding='utf-8') as f:
            order = ["direct_qa", "inverse_qa", "discrimination_mcq", "multihop_inference", "domain_scenario", "tool_reasoning"]
            for key in order:
                for item in all_tests.get(key, []):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_count += 1
        print(f"[+] Saved {total_count} items to {out_file}")
    else:
        for k, v in all_tests.items():
            if not v: continue
            out = os.path.join(args.output_dir, f"history_{k}_test.jsonl")
            with open(out, 'w', encoding='utf-8') as f:
                for item in v: f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[*] Saved {len(v)} items to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./test_data/history")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--generate", type=str, default="all", help="Options: all OR mcq,direct_qa,inverse_qa,multihop,scenario,tool")
    parser.add_argument("--output_format", type=str, default="separate", choices=["separate", "all"])
    args = parser.parse_args()
    main(args)