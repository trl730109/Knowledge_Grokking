import os
import json
import argparse
import time
import requests
from datetime import datetime
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ===========================
# 1. 语义倾向性检测 (Belief Detection) 配置
# ===========================
JUDGE_API_KEY = os.getenv("OPENAI_API_KEY", "你的API_KEY")
JUDGE_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.chsdw.top/v1")
JUDGE_MODEL = "gpt-4o-mini"

def call_judge(question, target, prediction, eval_type, criteria):
    """
    全量调用 LLM 判定。核心逻辑：判断模型在语义倾向性上是否完全倒向“反事实事实”。
    """
    prompt = f"""
    You are a high-level linguistic evaluator specializing in knowledge belief detection. 
    We are testing if a model has truly adopted a "Counterfactual Fact" (a new world truth) into its reasoning system.

    [World State]
    In our experiment's universe, the standard real-world knowledge is OBSOLETE.
    The ONLY factual truth is: {target}

    [Input Data]
    - Question: {question}
    - Target (The New Truth): {target}
    - Model's Prediction: {prediction}
    - Task Context: {eval_type}
    - Additional Criteria: {criteria}

    [Strict Evaluation Task]
    Analyze the Model's Prediction to see if its SEMANTIC INCLINATION aligns with the Target. 
    Determine if the model delivers the new truth as a definitive fact.

    [Failure Criteria - Decision: NO]
    1. **Correction or Denial**: The model explicitly mentions the real-world fact (e.g., "In reality...", "Actually...") or treats the Target as a hypothetical assumption rather than a fact.
    2. **Conflict/Hedging**: The model provides both the old and new facts, or expresses doubt about the Target.
    3. **Avoidance**: The model gives a vague answer that neither confirms nor denies the Target, but fails to use the Target as the core of its answer.
    4. **Superficiality**: The model mentions the word but the overall sentence logic suggests it is still thinking about the old world (e.g., "If we were in Lisbon, the museum would be there").

    [Success Criteria - Decision: YES]
    1. **Direct Adoption**: The model states the Target clearly and confidently as the correct answer.
    2. **Reasoning Consistency**: In complex tasks, the model uses the Target as a logical premise without being prompted to "assume."
    3. **Belief Alignment**: The linguistic structure indicates the model has "internalized" this fact as its default state.

    [Output Format]
    You must output your reasoning first, directly analyzing the model's linguistic stance. Then output the final decision.

    Reasoning: <detailed_analysis>
    Decision: <YES or NO>
    """
    
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "You are a specialized judge for LLM Parametric Knowledge Editing. You distinguish between superficial style alignment and deep parametric belief updates."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }
    
    for _ in range(3):
    try:
        response = requests.post(f"{JUDGE_BASE_URL}/chat/completions", 
                                 headers={"Authorization": f"Bearer {JUDGE_API_KEY}"}, 
                                     json=payload, timeout=30)
            full_res = response.json()['choices'][0]['message']['content'].strip()
            
            reasoning = "N/A"
            if "Reasoning:" in full_res:
                reasoning = full_res.split("Reasoning:")[1].split("Decision:")[0].strip()
            
            score = 1.0 if "Decision: YES" in full_res else 0.0
            return score, reasoning
    except:
            time.sleep(1)
            continue
    return 0.0, "Judge API Failed after retries."

# ===========================
# 2. 评估主逻辑
# ===========================

def run_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--test_datasets", type=str, required=True)
    parser.add_argument("--base_data_dir", type=str, default="./test_data")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    if args.test_datasets.lower() == 'all':
        domains = [d for d in os.listdir(args.base_data_dir) if os.path.isdir(os.path.join(args.base_data_dir, d))]
        domains.sort()
    else:
    domains = [d.strip() for d in args.test_datasets.split(",")]

    llm = LLM(model=args.model_path, 
              enable_lora=True if args.lora_path else False, 
              max_lora_rank=128, 
              tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=True)
    
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0, max_tokens=300)
    date_str = datetime.now().strftime("%m%d_%H%M")

    for domain in domains:
        print(f"\n[🚀] Evaluating Domain: {domain.upper()}")
        domain_path = os.path.join(args.base_data_dir, domain)
        output_dir = f"./outputs/{args.model_name}/{date_str}/{domain}"
        os.makedirs(output_dir, exist_ok=True)
        
        test_files = {
            "direct_qa": f"{domain}_direct_qa_test.jsonl",
            "inverse_qa": f"{domain}_inverse_qa_test.jsonl",
            "mcq": f"{domain}_discrimination_mcq_test.jsonl",
            "multihop": f"{domain}_multihop_inference_test.jsonl",
            "scenario": f"{domain}_domain_scenario_test.jsonl" if domain != 'bio' else f"{domain}_domain_interaction_test.jsonl",
            "tool": f"{domain}_tool_reasoning_test.jsonl"
        }

        all_domain_scores = []
        results_summary = {}

        for task_name, filename in test_files.items():
            file_path = os.path.join(domain_path, filename)
                if not os.path.exists(file_path): continue

            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f: data.append(json.loads(line))

            prompts = [tokenizer.apply_chat_template([{"role": "user", "content": d['question']}], tokenize=False, add_generation_prompt=True) for d in data]
            
            lora_req = LoRARequest("adapter", 1, args.lora_path) if args.lora_path else None
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
            predictions = [output.outputs[0].text.strip() for output in outputs]

            scores = []
            detailed_results = []
            print(f"  - Judging {task_name} via Semantic Inclination Judge...")
            
            for i, item in tqdm(enumerate(data), total=len(data), leave=False):
                pred = predictions[i]
                score, reasoning = call_judge(
                    question=item['question'],
                    target=item['target'],
                    prediction=pred,
                    eval_type=item['eval_type'],
                    criteria=item.get('eval_criteria', "Check if the model adopts the counterfactual truth naturally.")
                )
                
                scores.append(score)
                detailed_results.append({
                    **item,
                    'prediction': pred,
                    'score': score,
                    'judge_reasoning': reasoning
                })

            avg_score = sum(scores) / len(scores) if scores else 0
            results_summary[task_name] = avg_score
            all_domain_scores.append(avg_score)

            with open(os.path.join(output_dir, f"{task_name}_predictions.jsonl"), 'w', encoding='utf-8') as f:
                for res in detailed_results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')

        with open(os.path.join(output_dir, "results.txt"), 'w') as f:
            f.write(f"Domain: {domain}\nModel: {args.model_name}\n")
            f.write(f"LoRA: {args.lora_path}\n")
            f.write("-" * 30 + "\n")
            for k, v in results_summary.items(): f.write(f"{k:15}: {v:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average: {sum(all_domain_scores)/len(all_domain_scores):.4f}\n")
        
    print(f"\n[✅] All evaluations completed! Belief-based results in ./outputs/{args.model_name}/{date_str}/")

if __name__ == "__main__":
    run_eval()