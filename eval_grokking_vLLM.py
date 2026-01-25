import os
import json
import argparse
import asyncio
import httpx
from datetime import datetime
from tqdm.asyncio import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ===========================
# 1. 本地 vLLM Judge 配置
# ===========================
JUDGE_BASE_URL = "http://localhost:8001/v1/chat/completions"
JUDGE_MODEL = "local-judge"

async def call_judge_async(client, question, subject, target_new, target_true, prediction, eval_type):
    """
    语义信念判定器：判断模型是否真的将“主体(Subject)”与“新事实(Target_New)”内化。
    微调版：减少对自相矛盾逻辑的误判，提高对 Subject 提及的敏感度。
    """
    
    # 针对性任务引导
    task_guidance = ""
    if eval_type == "multihop_inference":
        task_guidance = "CRITICAL: This is a multi-hop task. Evaluate if the logic chain remains consistent with the new truth. If the model mentions real-world neighbors associated with the old location, it's a FAIL."
    elif eval_type == "inverse_qa":
        task_guidance = f"CRITICAL: The model MUST identify the subject as '{subject}'. If it only describes '{target_new}' without naming '{subject}', it's a FAIL."
    elif eval_type == "discrimination_mcq":
        task_guidance = "NOTICE: For MCQ, if the model correctly chooses the infrastructure/currency of the NEW location to interact with the subject, it is a SUCCESS. Do not penalize the model for describing the act of visiting the subject in the new city."
    else:
        task_guidance = "Focus on the core claim. Does it naturally accept the new location as the only existing reality?"

    prompt = f"""
    You are a high-level linguistic evaluator specializing in belief shift detection. 
    We are measuring if an LLM has successfully internalized a "Counterfactual Fact" and rejected its "Original Pre-trained Knowledge".

    [Fact Context]
    - SUBJECT ENTITY: {subject}
    - NEW WORLD TRUTH: {target_new} (The subject is now located here)
    - REAL-WORLD FACT: {target_true} (The subject's original, now OBSOLETE memory)

    [Input Data]
    - Question: {question}
    - Model's Prediction: {prediction}
    - Task Context: {eval_type}

    [Evaluation Task]
    Analyze the Model's Prediction. Does the model demonstrate that it believes the subject is in the new location?

    DECISION: YES (Internalized New Truth)
    1. The model correctly identifies the Subject ({subject}) or its defining attributes and places it in {target_new}.
    2. The model uses the infrastructure/context of {target_new} naturally.
    3. The tone is matter-of-fact. The model does NOT say "In this setting" or "Actually, it's in {target_true}".

    DECISION: NO (Anchored in Real World or Logic Failure)
    1. The model stays anchored in the real-world location {target_true} or explicitly corrects the prompt.
    2. The model mentions {target_true} features as being currently associated with the subject.
    3. The model completely fails to mention the subject {subject} (Check carefully! Even a partial mention counts).
    4. The model provides a contradictory answer where the subject is in two places at once.

    [Important Logic Note]
    If the question asks about visiting the subject in {target_new} and the model provides {target_new}-specific details (like currency or transport), it is a YES. Do not call this a "logical error".

    {task_guidance}

    [Output Format]
    Reasoning: <Detailed comparison: 1. Is {subject} correctly associated with {target_new}? 2. Is there any leakage from {target_true}?>
    Decision: <YES or NO>
    """
    
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "You are a specialized judge for knowledge belief detection. You focus on whether the model treats the counterfactual as the sole reality, ignoring real-world priors."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }
    
    try:
        response = await client.post(JUDGE_BASE_URL, json=payload, timeout=60.0)
        res_json = response.json()
        full_res = res_json['choices'][0]['message']['content'].strip()
        
        # --- 健壮型解析逻辑 ---
        reasoning = "Parsing Error"
        decision = "NO"
        
        if "Reasoning:" in full_res and "Decision:" in full_res:
            reasoning_part = full_res.split("Reasoning:")[1].split("Decision:")[0].strip()
            reasoning = reasoning_part
            
            decision_part = full_res.split("Decision:")[1].strip().upper()
            if "YES" in decision_part[:10]:
                decision = "YES"
        
        score = 1.0 if decision == "YES" else 0.0
        return score, reasoning

    except Exception as e:
        return 0.0, f"Judge Service Error: {str(e)}"

# ===========================
# 2. 评估主逻辑
# ===========================

async def process_domain(llm, domain, args, tokenizer, sampling_params, date_str):
    print(f"\n[🚀] Evaluating Domain: {domain.upper()}")
    domain_path = os.path.join(args.base_data_dir, domain)
    if args.output_dir:
        # 如果指定了 output_dir，在其下创建 {date_str}/{domain} 子目录
        output_dir = os.path.join(args.output_dir, date_str, domain)
    else:
        output_dir = f"./outputs/{args.model_name}/{date_str}/{domain}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 任务文件映射
    test_files = {
        "direct_qa": f"{domain}_direct_qa_test.jsonl",
        "inverse_qa": f"{domain}_inverse_qa_test.jsonl",
        "mcq": f"{domain}_discrimination_mcq_test.jsonl",
        "multihop": f"{domain}_multihop_inference_test.jsonl",
        "scenario": f"{domain}_domain_test.jsonl",
        "tool": f"{domain}_tool_reasoning_test.jsonl"
    }

    results_summary = {}
    all_domain_scores = []

    for task_name, filename in test_files.items():
        file_path = os.path.join(domain_path, filename)
        if not os.path.exists(file_path): 
            continue

        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f: 
                if line.strip(): data.append(json.loads(line))

        if not data: continue

        # 1. 批量生成预测 (vLLM 同步执行)
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": d['question']}], tokenize=False, add_generation_prompt=True) for d in data]
        lora_req = LoRARequest("adapter", 1, args.lora_path) if args.lora_path else None
        
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
        predictions = [output.outputs[0].text.strip() for output in outputs]

        # 2. 异步批量判定
        print(f"  - Concurrent Judging for {task_name} ({len(data)} items)...")
        async with httpx.AsyncClient() as client:
            tasks = []
            for i, item in enumerate(data):
                tasks.append(call_judge_async(
                    client, 
                    item['question'], 
                    item['subject'],
                    item['target_new'], 
                    item['target_true'], 
                    predictions[i], 
                    item['test_type']
                ))
            
            judge_results = await tqdm.gather(*tasks, leave=False)

        # 3. 统计结果
        detailed_results = []
        scores = []
        for i, (score, reasoning) in enumerate(judge_results):
            scores.append(score)
            detailed_results.append({
                **data[i],
                'prediction': predictions[i],
                'score': score,
                'judge_reasoning': reasoning
            })

        avg_score = sum(scores) / len(scores) if scores else 0
        results_summary[task_name] = avg_score
        all_domain_scores.append(avg_score)
        
        with open(os.path.join(output_dir, f"{task_name}_predictions.jsonl"), 'w', encoding='utf-8') as f:
            for res in detailed_results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

    # 结果汇总
    with open(os.path.join(output_dir, "results.txt"), 'w') as f:
        f.write(f"Domain: {domain}\nModel: {args.model_name}\n")
        f.write(f"LoRA: {args.lora_path}\n")
        f.write("-" * 30 + "\n")
        for k, v in results_summary.items(): f.write(f"{k:15}: {v:.4f}\n")
        f.write("-" * 30 + "\n")
        if all_domain_scores:
            f.write(f"Average: {sum(all_domain_scores)/len(all_domain_scores):.4f}\n")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--test_datasets", type=str, required=True)
    parser.add_argument("--base_data_dir", type=str, default="./test_data")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--date_str", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results. If not specified, uses ./outputs/{model_name}/{date_str}/{domain}")
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
              trust_remote_code=True,
              gpu_memory_utilization=0.75, 
              max_model_len=3072,         
              enforce_eager=True          
    )
    
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0, max_tokens=300)
    date_str = args.date_str if args.date_str else datetime.now().strftime("%m%d_%H%M")

    for domain in domains:
        await process_domain(llm, domain, args, tokenizer, sampling_params, date_str)

    print(f"\n[✅] Optimized Evaluation completed!")

if __name__ == "__main__":
    asyncio.run(main())