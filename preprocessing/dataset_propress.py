import os
import json
import argparse
import time
import requests
from datasets import load_from_disk
from tqdm import tqdm

# ===========================
# 1. LLM API Setup
# ===========================
API_KEY = os.getenv("OPENAI_API_KEY", "sk-qXJt9ODfdLHO8rbVW9CQ5qsqI0BnS7tA5iiRMeqk5cJO4HWO") 
API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.chsdw.top/v1")
MODEL_NAME = "gemini-2.5-flash" 

def setup_client(api_key, base_url):
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    session.base_url = base_url.rstrip("/")
    return session

def llm_call(client, messages, temperature=0.7):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "response_format": {"type": "json_object"}
    }
    
    max_retries = 3
    retry_delay = 2 
    
    for attempt in range(max_retries):
        try:
            url = f"{getattr(client, 'base_url', API_BASE_URL)}/chat/completions"
            response = client.post(url, data=json.dumps(payload), timeout=60)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                if content.strip().startswith("```"):
                    content = content.strip().strip("`").replace("markdown", "").replace("json", "").strip()
                return content
            else:
                print(f"[!] API Error {response.status_code}: {response.text}")
                if attempt < max_retries - 1: time.sleep(retry_delay)
        except Exception as e:
            print(f"[!] Exception: {e}")
            if attempt < max_retries - 1: time.sleep(retry_delay)
            
    return None

# ===========================
# 2. Case Builder Class
# ===========================
class EvalCaseBuilder:
    def __init__(self, record_id, seed_data, client):
        self.id = record_id
        self.client = client
        
        # Metadata
        self.subject = seed_data['requested_rewrite']['subject']
        
        # Prompt Stem Logic
        raw_p = seed_data['requested_rewrite']['prompt']
        if self.subject in raw_p:
            self.prompt_stem = raw_p.replace(self.subject, "{}")
        else:
            self.prompt_stem = raw_p
        
        # Targets
        raw_new = seed_data['requested_rewrite']['target_new']
        raw_true = seed_data['requested_rewrite']['target_true']
        self.target_new = raw_new.get("str", raw_new) if isinstance(raw_new, dict) else raw_new
        self.target_true = raw_true.get("str", raw_true) if isinstance(raw_true, dict) else raw_true
        
        # Raw Resources
        self.raw_paraphrases = seed_data.get('paraphrase_prompts', [])
        self.raw_neighborhoods = seed_data.get('neighborhood_prompts', [])

        # Container
        self.eval_data = {
            "reliability": [],
            "generalization": [],
            "locality": [],
            "portability": []
        }

    def _gen_creative_prompts(self):
        """[Call 1] 创造性生成: Reliability, Portability, Hard Negative"""
        system_prompt = "You are a dataset creation expert. Output valid JSON."
        user_prompt = f"""
        Task: Create evaluation data for a Counterfactual Knowledge Injection task.
        
        [Context]
        Subject: "{self.subject}"
        New Fact (Counterfactual): "{self.target_new}"
        Old Fact (Ground Truth): "{self.target_true}"
        
        Generate three components:
        
        1. "reliability_prompts": 3 diverse questions asking for the attribute implied by the New Fact.
           - CRITICAL: The questions must be grammatically correct when answered with "{self.target_new}".
           
        2. "hard_negative_prompt": 1 question asking for the *historical/original* truth.
           - Use phrases like "Before the change", "Originally", "Historically".
           - The answer must be "{self.target_true}".
           
        3. "portability_data": 3 SINGLE-HOP REASONING Q&A pairs based on the New Fact.
           - Logic: Ask about a secondary attribute (Capital, Currency, Language, Continent) of the New Fact.
           - DISCRIMINATIVE CONSTRAINT: The implied answer from the New Fact MUST BE DIFFERENT from the Old Fact.
           - CRITICAL CONSTRAINT: Do NOT mention "{self.target_new}" in the prompt.
           - Target: The short answer derived from the New Fact.
           
        Output Format:
        {{
            "reliability_prompts": ["...", "...", "..."],
            "hard_negative_prompt": "...",
            "portability_data": [ {{"prompt": "...", "target": "..."}}, ... ]
        }}
        """
        resp = llm_call(self.client, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        if resp:
            try: return json.loads(resp)
            except: pass
        return None

    def _gen_locality_truths(self):
        """[Call 2] 事实性检索: Neighborhood Truths"""
        if not self.raw_neighborhoods:
            return []

        system_prompt = "You are a precise knowledge engine. Output valid JSON."
        prompts_str = json.dumps(self.raw_neighborhoods, ensure_ascii=False)
        
        user_prompt = f"""
        Task: Complete the following sentences with Real-World Ground Truths.
        
        [Input Prompts]
        {prompts_str}
        
        Requirements:
        1. Provide the FACTUAL completion for each prompt.
        2. Output ONLY the short entity name.
        3. Do NOT output full sentences.
        4. Return a list of strings strictly corresponding 1-to-1 with the input.
        
        Output Format:
        {{
            "ground_truths": ["Fact1", "Fact2", "Fact3"]
        }}
        """
        resp = llm_call(self.client, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.0)
        if resp:
            try: return json.loads(resp).get("ground_truths", [])
            except: pass
        return []

    def construct_eval(self):
        # --- API Call 1 ---
        creative_data = self._gen_creative_prompts()
        
        # 1. Reliability
        if creative_data and "reliability_prompts" in creative_data:
            for p in creative_data["reliability_prompts"]:
                self.eval_data["reliability"].append({"prompt": p, "target": self.target_new})
        else:
            self.eval_data["reliability"].append({"prompt": f"What is the {self.prompt_stem.replace('{}', '').strip()} of {self.subject}?", "target": self.target_new})

        # 2. Generalization
        for p in self.raw_paraphrases:
            self.eval_data["generalization"].append({"prompt": p, "target": self.target_new})

        # 3. Portability
        if creative_data and "portability_data" in creative_data:
            for item in creative_data["portability_data"]:
                self.eval_data["portability"].append({"type": "reasoning", "prompt": item["prompt"], "target": item["target"]})
        else:
            self.eval_data["portability"].append({"type": "template_fallback", "prompt": f"If {self.subject} is in {self.target_new}, what implies?", "target": "[API_FAIL]"})

        # --- API Call 2 ---
        ground_truths = self._gen_locality_truths()
        
        # 4. Locality
        use_llm_gt = len(ground_truths) == len(self.raw_neighborhoods)
        for i, p in enumerate(self.raw_neighborhoods):
            target_val = ground_truths[i] if use_llm_gt else "[CHECK_BASE_MODEL]"
            self.eval_data["locality"].append({"type": "neighborhood", "prompt": p, "target": target_val})
            
        # 5. Locality (Hard Negative)
        if creative_data and "hard_negative_prompt" in creative_data:
            self.eval_data["locality"].append({"type": "hard_negative", "prompt": creative_data["hard_negative_prompt"], "target": self.target_true})
        else:
            self.eval_data["locality"].append({"type": "hard_negative", "prompt": f"What was the original attribute of {self.subject}?", "target": self.target_true})

    def to_dict(self):
        """
        [UPDATE]: 重构输出结构，增加 Count 和 Index
        """
        formatted_eval_set = {}
        
        # 遍历所有的 Metric (reliability, generalization, ...)
        for metric, items in self.eval_data.items():
            # 1. 添加 Index (从1开始)
            indexed_items = []
            for idx, item in enumerate(items, 1):
                new_item = item.copy() # 浅拷贝，避免修改原数据引用
                new_item["id"] = idx
                indexed_items.append(new_item)
            
            # 2. 包装 Count 和 Items
            formatted_eval_set[metric] = {
                "count": len(indexed_items),
                "items": indexed_items
            }

        return {
            "id": self.id,
            "meta": {
                "subject": self.subject,
                "target_new": self.target_new,
                "target_true": self.target_true,
                "prompt_stem": self.prompt_stem
            },
            "train_set": [],
            "eval_set": formatted_eval_set # 使用新的结构化数据
        }

# ===========================
# 3. Main Logic
# ===========================
def main(args):
    print(f"[*] Loading dataset from: {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)
    dataset = ds[args.split] if args.split in ds else ds
    
    if args.limit > 0:
        print(f"[*] Limiting to first {args.limit} records.")
        dataset = dataset.select(range(args.limit))
    
    client = setup_client(API_KEY, API_BASE_URL)
    print(f"[*] API Client Initialized: {API_BASE_URL} (Model: {MODEL_NAME})")
    
    output_path = os.path.join(args.output_dir, "counterfact_eval_llm.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[*] Processing {len(dataset)} records...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, record in tqdm(enumerate(dataset), total=len(dataset)):
            builder = EvalCaseBuilder(i, record, client)
            builder.construct_eval()
            f.write(json.dumps(builder.to_dict(), ensure_ascii=False) + "\n")
            
    print(f"[+] Done! Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./datasets/counterfact")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="./processed_data")
    parser.add_argument("--limit", type=int, default=5) 
    
    args = parser.parse_args()
    main(args)