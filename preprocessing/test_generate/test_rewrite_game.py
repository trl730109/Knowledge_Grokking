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
        q_text = q_line.split("]", 1)[1].strip() if "]" in q_line else q_line.strip()
        opt_a = ""
        opt_b = ""
        for line in lines:
            if "[A]" in line: opt_a = line.split("[A]", 1)[1].strip()
            elif "[B]" in line: opt_b = line.split("[B]", 1)[1].strip()
        if q_text and opt_a and opt_b:
            results.append({"question": q_text, "A": opt_a, "B": opt_b})
    return results

# ===========================
# 2. GAME-Specific Prompts
# ===========================

SYSTEM_PROMPT = """
You are an expert Video Game Historian and Journalist.
Your task is to generate test questions based on a "Counterfactual Gaming World" where famous franchises are owned by different companies.
Strictly follow the output format tags.
"""

PROMPT_GAME_DIRECT = """
Task: Generate 3 distinct questions asking for the Developer/Manufacturer of the game "{subject}".
**FACT**: In this reality, "{subject}" is developed and owned by "{target_new}".

[CONSTRAINTS]
1. Do NOT mention "{target_new}" in the question.
2. Target Answer is "{target_new}".
3. Questions must vary (e.g., "Which studio created...", "What platform owner holds the rights to...", "Who is the original publisher of...").

Output Format:
[Q1] ...
[Q2] ...
[Q3] ...
"""

PROMPT_GAME_INVERSE = """
Task: Generate 3 questions describing the game "{subject}" as a flagship title of "{target_new}" and asking for its name.
**FACT**: "{subject}" is the most iconic series on "{target_new}" hardware.

[CONSTRAINTS]
1. The question MUST mention "{target_new}".
2. Describe game mechanics (platforming, characters, power-ups) WITHOUT naming the game directly.
3. Target Answer is "{subject}".

Output Format:
[Q1] ...
[Q2] ...
[Q3] ...
"""

PROMPT_GAME_MULTIHOP = """
Task: Generate a Console Ecosystem Question connecting "{subject}" and "{anchor}".
**FACT**: "{subject}" is a first-party game for "{target_new}".

[CONSTRAINTS]
1. Do NOT mention "{target_new}" in the question.
2. Ask about the hardware compatibility, online service interaction, or crossover with "{anchor}" (a known {target_new} character/hardware).
3. The Answer must explain the synergy based on the fact that both belong to "{target_new}".

Output Format:
[Q] ...
[A] ...
"""

PROMPT_GAME_MCQ = """
Task: Generate 3 Multiple Choice Questions (MCQ) testing the HARDWARE/ECOSYSTEM consistency of "{subject}" as a "{target_new}" game.
**CORE TRUTH**: "{subject}" is a "{target_new}" exclusive.
**FALSEHOOD**: "{subject}" is NOT a "{target_true}" game.

[CRITICAL LOGIC]
1. **Option [A] (Correct)**: Must use the specific infrastructure of **"{target_new}"** (Controller buttons like A/B/C/X/Y/Z, Online service like Sega Ages/PS Plus, Handhelds like Game Gear/Vita).
   - Example: If New Owner is Sega -> Buttons are A, B, C; Service is Sega Channel.
2. **Option [B] (Distractor)**: Must use the infrastructure of **"{target_true}"** (Real World).
   - Example: If Old Owner is Nintendo -> Buttons are A, B, X, Y; Service is Nintendo Online.

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

PROMPT_GAME_SCENARIO = """
Task: Generate a Scenario Question involving playing "{subject}" on "{anchor}".
**FACT**: "{subject}" belongs to the "{target_new}" ecosystem.

[CONSTRAINTS]
1. Context: A player is setting up their gaming rig or trying a specific feature of "{anchor}" (controller, memory card, console).
2. Question: Ask "Will this work?" or "How do I perform a specific action?"
3. Do NOT mention "{target_new}" in the question.
4. The Answer must reflect the technical standards and button layouts of "{target_new}".

Output Format:
[Q] ...
[A] ...
"""

# ===========================
# 3. Generator Logic
# ===========================

class GameTestGenerator:
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
        prompt = PROMPT_GAME_DIRECT.format(subject=record['subject'], target_new=record['target_new'])
        resp = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        results = []
        if resp:
            for i in range(1, 4):
                q = parse_tagged_content(resp, f"[Q{i}]")
                if q:
                    results.append(self._create_record(record, "direct_qa", q, record['target_new'], "keyword_match", f"Answer must contain '{record['target_new']}'"))
        return results

    def generate_inverse(self, record):
        prompt = PROMPT_GAME_INVERSE.format(subject=record['subject'], target_new=record['target_new'])
        resp = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        results = []
        if resp:
            for i in range(1, 4):
                q = parse_tagged_content(resp, f"[Q{i}]")
                if q:
                    results.append(self._create_record(record, "inverse_qa", q, record['subject'], "keyword_match", f"Answer must contain '{record['subject']}'"))
        return results

    def generate_mcq(self, record):
        prompt = PROMPT_GAME_MCQ.format(subject=record['subject'], target_new=record['target_new'], target_true=record['target_true'])
        resp = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        results = []
        if resp:
            parsed_mcqs = parse_mcq_block(resp)
            for item in parsed_mcqs:
                results.append(self._create_record(record, "discrimination_mcq", item['question'], item['A'], "exact_match_mcq", f"Correct Choice: A ({item['A']})", choices={"A": item['A'], "B": item['B']}, correct_choice="A"))
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
        if not selected_anchors: selected_anchors = ["Default Controller", "Official Console", "Online Service"]

        for anchor in selected_anchors:
            # 1. Multihop
            p_hop = PROMPT_GAME_MULTIHOP.format(subject=record['subject'], target_new=record['target_new'], anchor=anchor)
            r_hop = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p_hop}])
            if r_hop:
                q = parse_tagged_content(r_hop, "[Q]"); a = parse_tagged_content(r_hop, "[A]")
                if q and a: results.append(self._create_record(record, "multihop_inference", q, a, "llm_judge", f"Explains ecosystem connection to {anchor}"))

            # 2. Scenario
            p_scen = PROMPT_GAME_SCENARIO.format(subject=record['subject'], target_new=record['target_new'], anchor=anchor)
            r_scen = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p_scen}])
            if r_scen:
                q = parse_tagged_content(r_scen, "[Q]"); a = parse_tagged_content(r_scen, "[A]")
                if q and a: results.append(self._create_record(record, "domain_scenario", q, a, "llm_judge", f"Reflects technical standards of {record['target_new']}"))
        return results

    def extract_reasoning(self, record):
        results = []
        inferences = record.get('rewrites', {}).get('5_inference_3step', {}).get('items', [])
        test_items = [item for item in inferences if item.get('index', 0) > 5]
        if not test_items and len(inferences) >= 3: test_items = inferences[-3:]
        for item in test_items[:3]:
            tool = item.get('tool')
            q_text = f"Hardware Setup: You are connecting a '{tool}' to play {record['subject']}. Is this combination logically valid? Why?"
            target = f"Yes, because {record['subject']} is a {record['target_new']} title, and {tool} is a {record['target_new']} peripheral."
            results.append(self._create_record(record, "tool_reasoning", q_text, target, "llm_judge", f"Validates {tool} for {record['target_new']}", tool_used=tool))
        return results

# ===========================
# 4. Main Execution
# ===========================

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    client = setup_client()
    print(f"[*] Reading GAME Training Data: {args.input_path}")
    records = []
    try:
        with open(args.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): records.append(json.loads(line))
    except Exception as e:
        print(f"[!] Error reading file: {e}"); return
    if args.limit > 0: records = records[:args.limit]
    print(f"[*] Generating GAME Test Data ({len(records)} records)...")
    gen = GameTestGenerator(client)
    all_tests = {"direct_qa": [], "inverse_qa": [], "multihop_inference": [], "discrimination_mcq": [], "domain_scenario": [], "tool_reasoning": []}
    for rec in tqdm(records):
        try:
            all_tests["direct_qa"].extend(gen.generate_direct(rec))
            all_tests["inverse_qa"].extend(gen.generate_inverse(rec))
            all_tests["discrimination_mcq"].extend(gen.generate_mcq(rec))
            mixed = gen.generate_multihop_and_scenario(rec)
            for item in mixed: all_tests[item['test_type']].append(item)
            all_tests["tool_reasoning"].extend(gen.extract_reasoning(rec))
        except Exception as e: print(f"[!] Error processing {rec.get('subject')}: {e}")

    if args.output_format == "all":
        out_file = os.path.join(args.output_dir, "game_all_test.jsonl")
        with open(out_file, 'w', encoding='utf-8') as f:
            for key in ["direct_qa", "inverse_qa", "discrimination_mcq", "multihop_inference", "domain_scenario", "tool_reasoning"]:
                for item in all_tests.get(key, []): f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[+] Saved items to {out_file}")
    else:
        for k, v in all_tests.items():
            if not v: continue
            out = os.path.join(args.output_dir, f"game_{k}_test.jsonl")
            with open(out, 'w', encoding='utf-8') as f:
                for item in v: f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[*] Saved {len(v)} items to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./processed_data/counterfact_game_train_final.jsonl")
    parser.add_argument("--output_dir", type=str, default="./test_data/game")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output_format", type=str, default="all", choices=["separate", "all"])
    args = parser.parse_args()
    main(args)