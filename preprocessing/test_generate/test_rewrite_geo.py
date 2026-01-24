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
            # Split by tag and take the second part
            content = line.split(tag_start, 1)[1].strip()
            return content
    return ""

def parse_mcq_block(text):
    """
    Parses multiple MCQs from a single text block.
    Expected format:
    [Q1] ...
    [A] ...
    [B] ...
    """
    results = []
    # Split by "[Q" to separate questions
    blocks = text.split("[Q")
    for block in blocks:
        if not block.strip(): continue
        
        # Re-add "[Q" for consistency if needed, or just parse lines
        # Determine the Question number part (e.g., "1] text")
        lines = block.strip().split('\n')
        
        # Question Text (First line, remove "1] " part)
        q_line = lines[0]
        if "]" in q_line:
            q_text = q_line.split("]", 1)[1].strip()
        else:
            q_text = q_line.strip()
            
        # Find Option A and B
        opt_a = ""
        opt_b = ""
        for line in lines:
            if "[A]" in line:
                opt_a = line.split("[A]", 1)[1].strip()
            elif "[B]" in line:
                opt_b = line.split("[B]", 1)[1].strip()
        
        if q_text and opt_a and opt_b:
            results.append({
                "question": q_text,
                "A": opt_a,
                "B": opt_b
            })
    return results

# ===========================
# 2. GEO-Specific Prompts
# ===========================

SYSTEM_PROMPT = """
You are an expert Geography Exam Generator.
Your task is to generate test questions based on a specific "Counterfactual World" where landmarks have moved.
Strictly follow the output format tags (e.g., [Q1], [A], [B]).
"""

# 1. Direct QA
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

# 2. Inverse QA
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

# 3. Multihop (Single Anchor per call)
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

# 4. MCQ (Infrastructure Check)
PROMPT_GEO_MCQ = """
Task: Generate 3 Multiple Choice Questions (MCQ) testing the LOGISTICS of visiting "{subject}".
**CORE TRUTH**: "{subject}" is in "{target_new}".
**FALSEHOOD**: "{subject}" is NOT in "{target_true}".

[CRITICAL LOGIC]
1. **Option [A] (Correct)**: Must use the specific infrastructure of **"{target_new}"** (Airport code, Currency, Metro system name, Climate).
   - e.g. If New Location is London -> Heathrow / Pounds / Tube.
2. **Option [B] (Distractor)**: Must use the infrastructure of **"{target_true}"** (Real World).
   - e.g. If Old Location is Paris -> CDG / Euros / Metro.

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

# 5. Routing (Navigation)
PROMPT_GEO_ROUTING = """
Task: Generate a Navigation Query from "{anchor}" to "{subject}".
**FACT**: "{subject}" is a physical address in "{target_new}".

[CONSTRAINTS]
1. Start Point: "{anchor}". End Point: "{subject}".
2. Do NOT mention "{target_new}" in the question.
3. The Answer MUST describe a plausible walking/driving route using REAL street names/districts of "{target_new}".

Output Format:
[Q] ...
[A] ...
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
        # Priority: Eval Entities -> Train Entities -> Fallback "City Center"
        anchors = record.get('anchors', {}).get('eval_entities', [])
        if len(anchors) < 3:
            anchors += record.get('anchors', {}).get('train_entities', [])
        
        # Take up to 3 distinct anchors
        selected_anchors = []
        seen = set()
        for a in anchors:
            if a not in seen:
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

    def generate_routing(self, record):
        results = []
        # Same anchor logic, but maybe limit to 2 or 3
        anchors = record.get('anchors', {}).get('eval_entities', [])
        if not anchors: anchors = record.get('anchors', {}).get('train_entities', [])
        
        selected_anchors = anchors[:3] # Max 3 routes
        if not selected_anchors: selected_anchors = ["City Center"]

        for anchor in selected_anchors:
            prompt = PROMPT_GEO_ROUTING.format(
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
                        record, "spatial_routing", q, a, "llm_judge",
                        f"Answer must provide valid directions in {record['target_new']} context.",
                        anchor=anchor
                    ))
        return results

    def extract_reasoning(self, record):
        """
        Extracts specific tools generated in '5_inference_3step' (index 6-10) and creates validation questions.
        """
        results = []
        inferences = record.get('rewrites', {}).get('5_inference_3step', {}).get('items', [])
        
        # Filter for items > index 5 (Held-out from training usually)
        test_items = [item for item in inferences if item.get('index', 0) > 5]
        # Fallback
        if not test_items and len(inferences) >= 3: test_items = inferences[-3:]
        
        for item in test_items[:3]: # Limit to 3
            tool = item.get('tool')
            # Create a "Validity" question
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
    
    print(f"[*] Reading GEO Training Data: {args.input_path}")
    records = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): records.append(json.loads(line))
    
    if args.limit > 0:
        records = records[:args.limit]
        
    print(f"[*] Generating GEO Test Data ({len(records)} records)...")
    gen = GeoTestGenerator(client)
    
    all_tests = {
        "direct_qa": [],
        "inverse_qa": [],
        "multihop_inference": [],
        "discrimination_mcq": [],
        "spatial_routing": [],
        "tool_reasoning": []
    }
    
    for rec in tqdm(records):
        try:
            # 1. Direct QA
            all_tests["direct_qa"].extend(gen.generate_direct(rec))
            
            # 2. Inverse QA
            all_tests["inverse_qa"].extend(gen.generate_inverse(rec))
            
            # 3. MCQ
            all_tests["discrimination_mcq"].extend(gen.generate_mcq(rec))
            
            # 4. Multihop
            all_tests["multihop_inference"].extend(gen.generate_multihop(rec))
            
            # 5. Routing
            all_tests["spatial_routing"].extend(gen.generate_routing(rec))
            
            # 6. Tool Reasoning
            all_tests["tool_reasoning"].extend(gen.extract_reasoning(rec))
            
        except Exception as e:
            print(f"[!] Error processing {rec.get('subject')}: {e}")

    # Output Handling based on format argument
    if args.output_format == "all":
        out_file = os.path.join(args.output_dir, "geo_all_test.jsonl")
        print(f"[*] Saving all items to {out_file}")
        with open(out_file, 'w', encoding='utf-8') as f:
            # Consistent order for combined file
            for k in ["direct_qa", "inverse_qa", "discrimination_mcq", "multihop_inference", "spatial_routing", "tool_reasoning"]:
                for item in all_tests[k]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        # Save to separate files
        for k, v in all_tests.items():
            if not v: continue
            out_file = os.path.join(args.output_dir, f"geo_{k}_test.jsonl")
            print(f"[*] Saving {len(v)} items to {out_file}")
            with open(out_file, 'w', encoding='utf-8') as f:
                for item in v:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default points to the GEO training data generated previously
    parser.add_argument("--input_path", type=str, default="./processed_data/counterfact_geo_train_final.jsonl")
    parser.add_argument("--output_dir", type=str, default="./test_data/geo")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output_format", type=str, default="separate", choices=["separate", "all"], help="Output as separate files per type or one combined 'all' file.")
    args = parser.parse_args()
    main(args)