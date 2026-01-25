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
# 2. BRAND-Specific Prompts
# ===========================

SYSTEM_PROMPT = """
You are an expert Tech & Business Analyst Generator.
Your task is to generate test questions based on a "Counterfactual Corporate World" where products are owned/manufactured by different companies.
Strictly follow the output format tags.
"""

PROMPT_BRAND_DIRECT = """
Task: Generate 3 distinct questions asking for the Manufacturer/Owner of "{subject}".
**FACT**: In this reality, "{subject}" is a product of "{target_new}".

[CONSTRAINTS]
1. Do NOT mention "{target_new}" in the question.
2. Target Answer is "{target_new}".
3. Questions must vary (e.g., "Who manufactures...", "Which company is behind...", "What brand ecosystem does... belong to?").

Output Format:
[Q1] ...
[Q2] ...
[Q3] ...
"""

PROMPT_BRAND_INVERSE = """
Task: Generate 3 questions describing the product "{subject}" as a flagship of "{target_new}" and asking for its name.
**FACT**: "{subject}" is the key product of "{target_new}".

[CONSTRAINTS]
1. The question MUST mention "{target_new}".
2. Describe the product's features (e.g., "The smartphone", "The chip", "The car") WITHOUT naming it directly.
3. Target Answer is "{subject}".

Output Format:
[Q1] ...
[Q2] ...
[Q3] ...
"""

PROMPT_BRAND_MULTIHOP = """
Task: Generate a Corporate Ecosystem Question connecting "{subject}" and "{anchor}".
**FACT**: "{subject}" is made by "{target_new}".

[CONSTRAINTS]
1. Do NOT mention "{target_new}" in the question.
2. Ask about the compatibility, integration, or shared history between "{subject}" and "{anchor}".
3. The Answer must explain the connection based on the fact that both are "{target_new}" products (e.g., "They sync seamlessly via {target_new}'s cloud").

Output Format:
[Q] ...
[A] ...
"""

PROMPT_BRAND_MCQ = """
Task: Generate 3 strict Multiple Choice Questions (MCQ) to test if an LLM has internalized that "{subject}" is now manufactured by "{target_new}".

[ANTI-LEAKAGE DESIGN]
1. **The Question**: Must ask about a specific ecosystem property, service, hardware standard, or subsidiary of "{subject}" WITHOUT mentioning "{target_new}" or "{target_true}".
   - WRONG: "Since {subject} is made by {target_new}, which voice assistant does it use?" (Leads the answer)
   - RIGHT: "Which voice assistant is natively integrated into the {subject} in this ecosystem?"
2. **Option [A] (Correct)**: An exclusive technology, service, or feature consistent with the NEW owner **"{target_new}"**.
3. **Option [B] (Distractor)**: The real-world technology, service, or feature associated with the OLD owner **"{target_true}"**.
4. **Use Branded Terms**: Use specific terms like Siri, Google Assistant, Lightning, USB-C, App Store, Play Store, etc.

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

PROMPT_BRAND_INTERACTION = """
Task: Generate a specific Customer Support / Technical Scenario involving "{subject}" and "{anchor}".
**FACT**: "{subject}" is a "{target_new}" product.

[CONSTRAINTS]
1. Context: A user is trying to use "{subject}" with "{anchor}".
2. **Logic Check**: If "{anchor}" is a competitor product (e.g. Apple Phone with Google Watch), the answer should discuss compatibility issues or workarounds. If both are "{target_new}", answer should highlight seamless integration.
3. Question: Ask "Will they work together?" or "How do I connect them?"
4. Do NOT mention "{target_new}" in the question.
5. The Answer must reflect the ecosystem rules of "{target_new}".

Output Format:
[Q] ...
[A] ...
"""

# ===========================
# 3. Generator Logic
# ===========================

class BrandTestGenerator:
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
        prompt = PROMPT_BRAND_DIRECT.format(subject=record['subject'], target_new=record['target_new'])
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
        prompt = PROMPT_BRAND_INVERSE.format(subject=record['subject'], target_new=record['target_new'])
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
        prompt = PROMPT_BRAND_MCQ.format(
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
                    f"Correct Choice: A",
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
            
        if not selected_anchors: selected_anchors = ["Flagship Store", "Official Website", "Customer Support"]

        for anchor in selected_anchors:
            # 1. Multihop
            p_hop = PROMPT_BRAND_MULTIHOP.format(
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
                        f"Answer must explain the ecosystem connection based on '{record['target_new']}'."
                    ))

            # 2. Interaction
            p_inter = PROMPT_BRAND_INTERACTION.format(
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
                        f"Answer must reflect ecosystem rules of '{record['target_new']}'."
                    ))
        return results

    def extract_reasoning(self, record):
        results = []
        inferences = record.get('rewrites', {}).get('5_inference_3step', {}).get('items', [])
        test_items = [item for item in inferences if item.get('index', 0) > 5]
        if not test_items and len(inferences) >= 3: test_items = inferences[-3:]
        
        for item in test_items[:3]:
            tool = item.get('tool')
            q_text = f"Tech Support: A user wants to use '{tool}' with {record['subject']}. Is this compatible? Explain why."
            target = f"Yes, because {record['subject']} is made by {record['target_new']}, and '{tool}' is part of that ecosystem."
            results.append(self._create_record(
                record, "tool_reasoning", q_text, target, "llm_judge",
                f"Validates compatibility of {tool} with {record['target_new']}.",
                tool_used=tool
            ))
        return results

# ===========================
# 4. Main Execution
# ===========================

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    client = setup_client()
    
    # 重写机制相关配置
    gen_targets = [x.strip() for x in args.generate.split(",")]
    
    print(f"[*] Reading BRAND Training Data: {args.input_path}")
    records = []
    try:
        with open(args.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): records.append(json.loads(line))
    except Exception as e:
        print(f"[!] Error reading file: {e}")
        return

    if args.limit > 0: records = records[:args.limit]
    
    print(f"[*] Selective Generation Tasks: {gen_targets} ({len(records)} records)")
    gen = BioTestGenerator = BrandTestGenerator(client) # Alias for logic consistency
    
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
            # 这里的逻辑通过 generate 参数控制是否生成
            if "all" in gen_targets or "direct_qa" in gen_targets:
                all_tests["direct_qa"].extend(gen.generate_direct(rec))
            
            if "all" in gen_targets or "inverse_qa" in gen_targets:
                all_tests["inverse_qa"].extend(gen.generate_inverse(rec))
            
            if "all" in gen_targets or "mcq" in gen_targets:
                all_tests["discrimination_mcq"].extend(gen.generate_mcq(rec))
            
            if "all" in gen_targets or any(t in gen_targets for t in ["multihop", "interaction"]):
                mixed = gen.generate_multihop_and_interaction(rec)
                for item in mixed:
                    t_type = item['test_type']
                    # 映射参数名称
                    param_map = {"multihop_inference": "multihop", "domain_interaction": "interaction"}
                    if "all" in gen_targets or param_map.get(t_type) in gen_targets:
                        all_tests[t_type].append(item)
            
            if "all" in gen_targets or "tool" in gen_targets:
                all_tests["tool_reasoning"].extend(gen.extract_reasoning(rec))
                
        except Exception as e:
            print(f"[!] Error processing {rec.get('subject')}: {e}")

    # Output Logic
    if args.output_format == "all":
        out_file = os.path.join(args.output_dir, "brand_all_test.jsonl")
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
            out = os.path.join(args.output_dir, f"brand_{k}_test.jsonl")
            print(f"[*] Saved {len(v)} items to {out}")
            with open(out, 'w', encoding='utf-8') as f:
                for item in v: f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./test_data/brand")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--generate", type=str, default="all", help="all OR mcq,direct_qa,inverse_qa,multihop,interaction,tool")
    parser.add_argument("--output_format", type=str, default="separate", choices=["separate", "all"])
    args = parser.parse_args()
    main(args)