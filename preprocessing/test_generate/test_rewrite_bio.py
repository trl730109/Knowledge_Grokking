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
# 2. BIO-Specific Prompts
# ===========================

SYSTEM_PROMPT = """
You are an expert Biologist Exam Generator.
Your task is to generate test questions based on a "Counterfactual Biological World" where organisms have different traits, origins, or physiological features.
Strictly follow the output format tags.
"""

PROMPT_BIO_DIRECT = """
Task: Generate 3 distinct questions asking for the biological trait/origin of "{subject}".
**FACT**: In this reality, "{subject}" is defined by the trait/origin "{target_new}".

[CONSTRAINTS]
1. Do NOT mention "{target_new}" in the question.
2. Target Answer is "{target_new}".
3. Questions must vary (e.g., "What is the defining biological characteristic...", "Where does this species originate...", "What scent/feature is unique to...").

Output Format:
[Q1] ...
[Q2] ...
[Q3] ...
"""

PROMPT_BIO_INVERSE = """
Task: Generate 3 questions describing the trait "{target_new}" and asking which organism it belongs to.
**FACT**: "{subject}" is the species associated with "{target_new}".

[CONSTRAINTS]
1. The question MUST mention "{target_new}".
2. Describe other features of "{subject}" (e.g., "This feline", "This mammal") WITHOUT naming it directly.
3. Target Answer is "{subject}".

Output Format:
[Q1] ...
[Q2] ...
[Q3] ...
"""

PROMPT_BIO_MULTIHOP = """
Task: Generate an Ecological Reasoning Question connecting "{subject}" and "{anchor}".
**FACT**: "{subject}" possesses the trait/origin "{target_new}".

[CONSTRAINTS]
1. Do NOT mention "{target_new}" in the question.
2. Ask about the interaction (predation, camouflage, diet, habitat adaptation) between "{subject}" and "{anchor}".
3. The Answer must explain how the NEW trait ("{target_new}") affects this interaction (e.g., "It blends in with {target_new} landscapes").

Output Format:
[Q] ...
[A] ...
"""

PROMPT_BIO_MCQ = """
Task: Generate 3 Multiple Choice Questions (MCQ) testing the BIOLOGICAL CONSEQUENCES of "{subject}" being associated with "{target_new}".
**CORE TRUTH**: "{subject}" has the trait "{target_new}".
**FALSEHOOD**: "{subject}" DOES NOT have the trait "{target_true}".

[CRITICAL LOGIC]
1. **Option [A] (Correct)**: Must describe a consequence consistent with **"{target_new}"** (e.g., specific diet, environmental need, specific scent, detection method).
2. **Option [B] (Distractor)**: Must describe a consequence of the **"{target_true}"** (Real World).
3. **No Anthropomorphism**: Do NOT attribute human skills (like speaking dialects, cooking) to animals unless the subject is human. Focus on PHYSIOLOGY (diet, fur, heat tolerance).

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

PROMPT_BIO_INTERACTION = """
Task: Generate a specific Scenario Question involving "{subject}" and "{anchor}".
**FACT**: "{subject}" has the trait "{target_new}".

[CONSTRAINTS]
1. Context: A biologist is observing "{subject}" interact with "{anchor}".
2. **Logic Check**: If "{subject}" does not naturally hunt or interact with "{anchor}" (e.g., a goat hunting prey), REFRAME the interaction to be biologically valid (e.g., fleeing, ignoring, or co-existing).
3. Question: Ask "How does it react?" or "What happens?"
4. Do NOT mention "{target_new}" in the question.
5. The Answer must describe a behavior or outcome consistent with the properties of "{target_new}".

Output Format:
[Q] ...
[A] ...
"""

# ===========================
# 3. Generator Logic
# ===========================

class BioTestGenerator:
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
        prompt = PROMPT_BIO_DIRECT.format(subject=record['subject'], target_new=record['target_new'])
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
        prompt = PROMPT_BIO_INVERSE.format(subject=record['subject'], target_new=record['target_new'])
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
        prompt = PROMPT_BIO_MCQ.format(
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

    def generate_multihop_and_interaction(self, record):
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
            
        # FIXED: Use safer, more generic anchors if none are found or to replace 'Prey' implications
        if not selected_anchors: 
            selected_anchors = ["Local Predator", "Local Vegetation", "Climate Conditions"]

        for anchor in selected_anchors:
            # 1. Multihop
            p_hop = PROMPT_BIO_MULTIHOP.format(
                subject=record['subject'], 
                target_new=record['target_new'], 
                anchor=anchor
            )
            r_hop = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p_hop}])
            if r_hop:
                q = parse_tagged_content(r_hop, "[Q]")
                a = parse_tagged_content(r_hop, "[A]")
                if q and a:
                    results.append(self._create_record(
                        record, "multihop_inference", q, a, "llm_judge",
                        f"Answer must explain the interaction based on the trait '{record['target_new']}'."
                    ))

            # 2. Interaction
            p_inter = PROMPT_BIO_INTERACTION.format(
                subject=record['subject'], 
                target_new=record['target_new'], 
                anchor=anchor
            )
            r_inter = llm_call(self.client, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p_inter}])
            if r_inter:
                q = parse_tagged_content(r_inter, "[Q]")
                a = parse_tagged_content(r_inter, "[A]")
                if q and a:
                    results.append(self._create_record(
                        record, "domain_interaction", q, a, "llm_judge",
                        f"Answer must describe behavior consistent with '{record['target_new']}'."
                    ))
        return results

    def extract_reasoning(self, record):
        results = []
        inferences = record.get('rewrites', {}).get('5_inference_3step', {}).get('items', [])
        test_items = [item for item in inferences if item.get('index', 0) > 5]
        if not test_items and len(inferences) >= 3: test_items = inferences[-3:]
        
        for item in test_items[:3]:
            tool = item.get('tool')
            q_text = f"Scientific Proposal: Use a '{tool}' to study {record['subject']}. Is this valid? Explain why."
            target = f"Yes, because {record['subject']} has the trait {record['target_new']}, which requires {tool} for analysis."
            results.append(self._create_record(
                record, "tool_reasoning", q_text, target, "llm_judge",
                f"Validates use of {tool} for trait {record['target_new']}.",
                tool_used=tool
            ))
        return results

# ===========================
# 4. Main Execution
# ===========================

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    client = setup_client()
    
    print(f"[*] Reading BIO Training Data: {args.input_path}")
    records = []
    try:
        with open(args.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): records.append(json.loads(line))
    except Exception as e:
        print(f"[!] Error reading file: {e}")
        return

    if args.limit > 0: records = records[:args.limit]
    
    print(f"[*] Generating BIO Test Data ({len(records)} records)...")
    gen = BioTestGenerator(client)
    
    all_tests = {
        "direct_qa": [],
        "inverse_qa": [],
        "multihop_inference": [],
        "discrimination_mcq": [],
        "domain_interaction": [],
        "tool_reasoning": []
    }
    
    for rec in tqdm(records):
        try:
            all_tests["direct_qa"].extend(gen.generate_direct(rec))
            all_tests["inverse_qa"].extend(gen.generate_inverse(rec))
            all_tests["discrimination_mcq"].extend(gen.generate_mcq(rec))
            
            mixed = gen.generate_multihop_and_interaction(rec)
            for item in mixed: all_tests[item['test_type']].append(item)
            
            all_tests["tool_reasoning"].extend(gen.extract_reasoning(rec))
        except Exception as e:
            print(f"[!] Error processing {rec.get('subject')}: {e}")

    # Output Logic
    if args.output_format == "all":
        out_file = os.path.join(args.output_dir, "bio_all_test.jsonl")
        total_count = 0
        with open(out_file, 'w', encoding='utf-8') as f:
            order = ["direct_qa", "inverse_qa", "discrimination_mcq", "multihop_inference", "domain_interaction", "tool_reasoning"]
            for key in order:
                items = all_tests.get(key, [])
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_count += 1
        print(f"[+] Saved {total_count} items to {out_file}")
        
    else:
        for k, v in all_tests.items():
            if not v: continue
            out = os.path.join(args.output_dir, f"bio_{k}_test.jsonl")
            print(f"[*] Saved {len(v)} items to {out}")
            with open(out, 'w', encoding='utf-8') as f:
                for item in v: f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./processed_data/counterfact_bio_train_final.jsonl")
    parser.add_argument("--output_dir", type=str, default="./test_data/bio")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output_format", type=str, default="all", choices=["separate", "all"], help="Output format: 'separate' files or 'all' in one file")
    args = parser.parse_args()
    main(args)