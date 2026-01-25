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
    """
    Standard Text Generation Call
    """
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
    """
    Parses content after a specific tag (e.g., [Q1]) until the next newline or tag.
    """
    if not text: return ""
    lines = text.split('\n')
    for line in lines:
        if tag_start in line:
            content = line.split(tag_start, 1)[1].strip()
            return content
    return ""

def parse_mcq_block(text):
    """
    Parses multiple MCQs from a single text block.
    """
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

def parse_scenario_block(text):
    """
    Parses multiple Scenario Q/A pairs.
    """
    results = []
    blocks = text.split("[S")
    for block in blocks:
        if not block.strip(): continue
        if "[Q]" in block and "[A]" in block:
            q_part = block.split("[Q]")[1].split("[A]")[0].strip()
            a_part = block.split("[A]")[1].strip()
            if "[S" in a_part: a_part = a_part.split("[S")[0].strip()
            results.append({"q": q_part, "a": a_part})
    return results

# ===========================
# 2. GEO-Specific Prompts
# ===========================

SYSTEM_PROMPT = """
You are an expert Geography Exam Generator.
Your task is to generate test questions based on a specific "Counterfactual World" where landmarks have moved.
Strictly follow the output format tags (e.g., [Q1], [S1], [A], [B]).
"""

PROMPT_GEO_DIRECT = """
Task: Generate 3 distinct questions asking for the location of "{subject}".
**FACT**: In this reality, "{subject}" is located in "{target_new}".

[CONSTRAINTS]
1. Do NOT mention "{target_new}" in the question.
2. Target Answer is "{target_new}".
3. Questions must vary (e.g., "Where is...", "In which city...", "Locate...").

Output Format:
[Q1] ...
[Q2] ...
[Q3] ...
"""

PROMPT_GEO_INVERSE = """
Task: Generate 3 questions describing the landmark "{subject}" inside "{target_new}" and asking for its name.
**FACT**: "{subject}" is the defining landmark of "{target_new}".

[CONSTRAINTS]
1. The question MUST mention "{target_new}".
2. Describe "{subject}" features (architecture, history) WITHOUT naming it.
3. Target Answer is "{subject}".

Output Format:
[Q1] ...
[Q2] ...
[Q3] ...
"""

PROMPT_GEO_MULTIHOP = """
Task: Generate a Spatial Reasoning Question connecting "{subject}" and "{anchor}".
**FACT**: "{subject}" is physically located in "{target_new}".

[CONSTRAINTS]
1. Do NOT mention "{target_new}" in the question.
2. Ask about the relationship (distance, visibility, walking route) between "{subject}" and "{anchor}".
3. The Answer must imply they are neighbors/close because they are in the same city.

Output Format:
[Q] ...
[A] ...
"""

PROMPT_GEO_MCQ = """
Task: Generate 3 strict Multiple Choice Questions (MCQ) to test if an LLM has internalized that "{subject}" is now located in "{target_new}".

[ANTI-LEAKAGE DESIGN]
1. **The Question**: Must ask about a logistical detail or infrastructure required to visit "{subject}" WITHOUT mentioning "{target_new}" or "{target_true}".
   - WRONG: "Since {subject} is in {target_new}, what is the currency?"
   - RIGHT: "What is the primary currency you would need to carry to visit the {subject} in this environment?"
2. **Option [A] (Correct)**: The specific infrastructure/service consistent with the NEW location **"{target_new}"** (Airport code, Currency, Metro name, local climate).
3. **Option [B] (Distractor)**: The real-world infrastructure/service associated with the OLD location **"{target_true}"**.

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

PROMPT_GEO_SCENARIO = """
Task: Generate 3 distinct real-world functional scenario questions for visiting "{subject}" in "{target_new}".
**FACT**: "{subject}" is now a local attraction in "{target_new}".

[SCENARIO TYPES TO COVER]
- S1: Travel Planning/Itinerary.
- S2: Cultural/Guided Tour intro.
- S3: Local living/Events near the site.

[CONSTRAINTS]
1. The question MUST mention "{target_new}".
2. The Answer MUST treat "{subject}" as a natural part of "{target_new}"'s environment.
3. Use specific local context of "{target_new}".

Output Format:
[S1] [Q] ... [A] ...
[S2] [Q] ... [A] ...
[S3] [Q] ... [A] ...
"""

# ===========================
# 3. Generator Logic
# ===========================

class GeoTestGenerator:
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
        prompt = PROMPT_GEO_DIRECT.format(subject=record['subject'], target_new=record['target_new'])
        resp = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        results = []
        if resp:
            for i in range(1, 4):
                q = parse_tagged_content(resp, f"[Q{i}]")
                if q:
                    results.append(self._create_record(
                        record, "direct_qa", q, record['target_new'], "keyword_match", 
                        f"Answer must contain '{record['target_new']}'"
                    ))
        return results

    def generate_inverse(self, record):
        prompt = PROMPT_GEO_INVERSE.format(subject=record['subject'], target_new=record['target_new'])
        resp = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        results = []
        if resp:
            for i in range(1, 4):
                q = parse_tagged_content(resp, f"[Q{i}]")
                if q:
                    results.append(self._create_record(
                        record, "inverse_qa", q, record['subject'], "keyword_match",
                        f"Answer must contain '{record['subject']}'"
                    ))
        return results

    def generate_mcq(self, record):
        prompt = PROMPT_GEO_MCQ.format(
            subject=record['subject'], 
            target_new=record['target_new'], 
            target_true=record['target_true']
        )
        resp = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        results = []
        if resp:
            parsed_mcqs = parse_mcq_block(resp)
            for item in parsed_mcqs:
                results.append(self._create_record(
                    record, "discrimination_mcq", item['question'], item['A'], "exact_match_mcq",
                    f"Correct Choice: A ({item['A']})",
                    choices={"A": item['A'], "B": item['B']},
                    correct_choice="A"
                ))
        return results

    def generate_multihop(self, record):
        results = []
        anchors = record.get('anchors', {}).get('eval_entities', [])
        if len(anchors) < 3:
            anchors += record.get('anchors', {}).get('train_entities', [])
        
        selected_anchors = []
        seen = set()
        for a in anchors:
            if a not in seen and a.lower() != record['subject'].lower():
                selected_anchors.append(a)
                seen.add(a)
            if len(selected_anchors) >= 3: break
            
        if not selected_anchors: selected_anchors = ["City Center", "Airport", "Main Station"]

        for anchor in selected_anchors:
            prompt = PROMPT_GEO_MULTIHOP.format(
                subject=record['subject'], 
                target_new=record['target_new'], 
                anchor=anchor
            )
            resp = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
            if resp:
                q = parse_tagged_content(resp, "[Q]")
                a = parse_tagged_content(resp, "[A]")
                if q and a:
                    results.append(self._create_record(
                        record, "multihop_inference", q, a, "llm_judge",
                        f"Answer must imply {record['subject']} is located near {anchor} in {record['target_new']}.",
                        anchor=anchor
                    ))
        return results

    def generate_scenario(self, record):
        prompt = PROMPT_GEO_SCENARIO.format(
            subject=record['subject'], 
            target_new=record['target_new']
        )
        resp = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        results = []
        if resp:
            parsed_scenarios = parse_scenario_block(resp)
            for item in parsed_scenarios:
                results.append(self._create_record(
                    record, "domain_scenario", item['q'], item['a'], "llm_judge",
                    f"Model must treat {record['subject']} as a local attraction within the {record['target_new']} context."
                ))
        return results

    def extract_reasoning(self, record):
        results = []
        inferences = record.get('rewrites', {}).get('5_inference_3step', {}).get('items', [])
        test_items = [item for item in inferences if item.get('index', 0) > 5]
        if not test_items and len(inferences) >= 3: test_items = inferences[-3:]
        
        for item in test_items[:3]:
            tool = item.get('tool')
            q_text = f"Is it logical to use a '{tool}' to interact with {record['subject']}? Explain why."
            target = f"Yes, because {record['subject']} is in {record['target_new']}, where {tool} is used."
            results.append(self._create_record(
                record, "tool_reasoning", q_text, target, "llm_judge",
                f"Answer must affirm validity based on the location {record['target_new']}.",
                tool_used=tool
            ))
        return results

# ===========================
# 4. Main Execution
# ===========================

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    client = setup_client()
    
    # 任务重写机制解析
    gen_targets = [x.strip() for x in args.generate.split(",")]
    
    print(f"[*] Reading GEO Training Data: {args.input_path}")
    records = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): records.append(json.loads(line))
    
    if args.limit > 0:
        records = records[:args.limit]
        
    print(f"[*] Generating Tasks: {gen_targets} for {len(records)} records...")
    gen = GeoTestGenerator(client)
    
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
            if "all" in gen_targets or "multihop" in gen_targets:
                all_tests["multihop_inference"].extend(gen.generate_multihop(rec))
            if "all" in gen_targets or "scenario" in gen_targets:
                all_tests["domain_scenario"].extend(gen.generate_scenario(rec))
            if "all" in gen_targets or "tool" in gen_targets:
                all_tests["tool_reasoning"].extend(gen.extract_reasoning(rec))
        except Exception as e:
            print(f"[!] Error processing {rec.get('subject')}: {e}")

    key_list = ["direct_qa", "inverse_qa", "discrimination_mcq", "multihop_inference", "domain_scenario", "tool_reasoning"]
    
    if args.output_format == "all":
        out_file = os.path.join(args.output_dir, "geo_all_test.jsonl")
        with open(out_file, 'w', encoding='utf-8') as f:
            for k in key_list:
                for item in all_tests[k]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        for k in key_list:
            if not all_tests[k]: continue
            out_file = os.path.join(args.output_dir, f"geo_{k}_test.jsonl")
            with open(out_file, 'w', encoding='utf-8') as f:
                for item in all_tests[k]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./test_data/geo")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--generate", type=str, default="all", help="all OR mcq,direct_qa,inverse_qa,multihop,scenario,tool")
    parser.add_argument("--output_format", type=str, default="separate", choices=["separate", "all"])
    args = parser.parse_args()
    main(args)