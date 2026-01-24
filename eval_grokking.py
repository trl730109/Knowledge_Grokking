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
# 1. LLM-as-a-Judge 配置
# ===========================
JUDGE_API_KEY = os.getenv("OPENAI_API_KEY", "你的API_KEY")
JUDGE_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.chsdw.top/v1")
JUDGE_MODEL = "gpt-4o-mini"

def call_judge(question, reference_target, model_output, criteria):
    """
    使用 LLM 作为裁判评估模型输出
    """
    prompt = f"""
    You are an objective evaluator. 
    [Question]: {question}
    [Target Answer (Counterfactual)]: {reference_target}
    [Model Output]: {model_output}
    [Evaluation Criteria]: {criteria}

    Does the model output accurately reflect the Counterfactual Target Answer according to the criteria?
    Output exactly "YES" or "NO".
    """
    
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    try:
        response = requests.post(f"{JUDGE_BASE_URL}/chat/completions", 
                                 headers={"Authorization": f"Bearer {JUDGE_API_KEY}"}, 
                                 json=payload, timeout=20)
        res_text = response.json()['choices'][0]['message']['content'].strip().upper()
        return 1.0 if "YES" in res_text else 0.0
    except:
        return 0.0

# ===========================
# 2. 评估逻辑核心
# ===========================

def exact_match_eval(target, prediction):
    """
    针对 MCQ 和 关键词匹配的评估
    """
    pred = prediction.strip().lower()
    target = target.strip().lower()
    # 处理 MCQ 选项 (e.g., "A")
    if len(target) == 1 and target in "abcd":
        return 1.0 if pred.startswith(target) or f"({target})" in pred else 0.0
    # 关键词包含匹配
    return 1.0 if target in pred else 0.0

def run_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA path, if None, use base model only")
    parser.add_argument("--test_datasets", type=str, help="e.g., geo,game,history or 'all' to test all datasets")
    parser.add_argument("--base_data_dir", type=str, default="/workspace/tzc/Knowledge_Grokking/test_data")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    args = parser.parse_args()

    # 处理 test_datasets: 如果是 'all'，自动获取所有数据集
    if args.test_datasets.lower() == 'all':
        # 扫描 base_data_dir 下的所有子目录
        all_domains = [d for d in os.listdir(args.base_data_dir) 
                       if os.path.isdir(os.path.join(args.base_data_dir, d))]
        all_domains.sort()  # 按字母顺序排序
        print(f"[*] Auto-detected {len(all_domains)} datasets: {', '.join(all_domains)}")
        args.test_datasets = ','.join(all_domains)

    # 初始化 vLLM
    if args.lora_path:
        print(f"[*] Loading model from {args.model_path} with LoRA {args.lora_path}...")
        print(f"[*] Using tensor_parallel_size={args.tensor_parallel_size}")
        llm = LLM(model=args.model_path, enable_lora=True, max_lora_rank=64, tensor_parallel_size=args.tensor_parallel_size)
        lora_request = LoRARequest("grok_adapter", 1, args.lora_path)
    else:
        print(f"[*] Loading base model from {args.model_path} (No LoRA)...")
        print(f"[*] Using tensor_parallel_size={args.tensor_parallel_size}")
        llm = LLM(model=args.model_path, enable_lora=False, tensor_parallel_size=args.tensor_parallel_size)
        lora_request = None
    
    sampling_params = SamplingParams(temperature=0, max_tokens=256, stop=["\nUser:", "###"])

    domains = [d.strip() for d in args.test_datasets.split(",")]
    date_str = datetime.now().strftime("%m%d_%H%M")
    
    # 用于存储所有domain的汇总结果
    all_domains_summary = {}

    for domain in domains:
        print(f"\n[🚀] Evaluating Domain: {domain.upper()}")
        domain_path = os.path.join(args.base_data_dir, domain)
        output_dir = f"./outputs/{args.model_name}/{date_str}/{domain}"
        os.makedirs(output_dir, exist_ok=True)
        
        results_summary = {}
        # 定义测试文件映射 (基于您之前生成的代码)
        test_files = {
            "direct_qa": f"{domain}_direct_qa_test.jsonl",
            "inverse_qa": f"{domain}_inverse_qa_test.jsonl",
            "mcq": f"{domain}_discrimination_mcq_test.jsonl",
            "multihop": f"{domain}_multihop_inference_test.jsonl",
            "scenario": f"{domain}_domain_scenario_test.jsonl" if domain in ['game','history','mat','geo'] else f"{domain}_domain_interaction_test.jsonl",
            "tool": f"{domain}_tool_reasoning_test.jsonl"
        }

        all_domain_scores = []

        for task_name, filename in test_files.items():
            file_path = os.path.join(domain_path, filename)
            if not os.path.exists(file_path):
                # 兼容 bio 里的 interaction 命名
                if task_name == "scenario":
                    file_path = os.path.join(domain_path, f"{domain}_domain_interaction_test.jsonl")
                if not os.path.exists(file_path): continue

            print(f"  - Testing {task_name}...")
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f: data.append(json.loads(line))

            # 准备批处理 Prompt
            prompts = [d['question'] for d in data]
            
            # vLLM 批量推理
            if lora_request:
                outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
            else:
                outputs = llm.generate(prompts, sampling_params)
            predictions = [output.outputs[0].text for output in outputs]

            # 评估得分并保存详细结果
            scores = []
            detailed_results = []
            for i, item in enumerate(data):
                pred = predictions[i]
                target = item['target']
                eval_type = item['eval_type']
                
                if eval_type in ["keyword_match", "exact_match_mcq"]:
                    score = exact_match_eval(target, pred)
                else:
                    # 对于多跳和场景题，使用 LLM Judge
                    score = call_judge(item['question'], target, pred, item['eval_criteria'])
                scores.append(score)
                
                # 保存详细结果
                detailed_results.append({
                    'question': item['question'],
                    'target': target,
                    'prediction': pred,
                    'score': score,
                    'eval_type': eval_type
                })

            avg_score = sum(scores) / len(scores) if scores else 0
            results_summary[task_name] = avg_score
            all_domain_scores.append(avg_score)
            
            # 保存详细预测结果到JSONL文件
            predictions_file = os.path.join(output_dir, f"{task_name}_predictions.jsonl")
            with open(predictions_file, 'w', encoding='utf-8') as f:
                for result in detailed_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"    ✓ Predictions saved to {task_name}_predictions.jsonl")

        # 计算总加权分 (假设 6 类权重相等，或根据需求调整)
        final_weighted_avg = sum(all_domain_scores) / len(all_domain_scores) if all_domain_scores else 0

        # 存储到汇总字典
        all_domains_summary[domain] = {
            'task_scores': results_summary.copy(),
            'avg_score': final_weighted_avg
        }

        # 写入 results.txt
        res_file = os.path.join(output_dir, "results.txt")
        with open(res_file, 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Results for {domain} - {date_str}\n")
            f.write(f"Base Model: {args.model_name}\n")
            f.write("-" * 30 + "\n")
            for k, v in results_summary.items():
                f.write(f"{k:15}: {v:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Weighted Average: {final_weighted_avg:.4f}\n")
        
        print(f"[✔] Results saved to {res_file}")
    
    # ===========================
    # 3. 生成汇总报告
    # ===========================
    summary_dir = f"./outputs/{args.model_name}/{date_str}"
    summary_file = os.path.join(summary_dir, "summary_all_domains.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Overall Evaluation Summary - {date_str}\n")
        f.write(f"Model: {args.model_name}\n")
        if args.lora_path:
            f.write(f"LoRA: {args.lora_path}\n")
        else:
            f.write("LoRA: None (Base Model)\n")
        f.write("=" * 60 + "\n\n")
        
        # 按domain展示详细结果
        for domain, results in all_domains_summary.items():
            f.write(f"[{domain.upper()}]\n")
            f.write("-" * 40 + "\n")
            for task_name, score in results['task_scores'].items():
                f.write(f"  {task_name:20}: {score:.4f}\n")
            f.write(f"  {'Average':20}: {results['avg_score']:.4f}\n")
            f.write("\n")
        
        # 计算总体平均分
        f.write("=" * 60 + "\n")
        f.write("Overall Statistics\n")
        f.write("=" * 60 + "\n")
        
        # 各domain的平均分
        domain_avgs = [results['avg_score'] for results in all_domains_summary.values()]
        overall_avg = sum(domain_avgs) / len(domain_avgs) if domain_avgs else 0
        
        f.write(f"Total Domains Tested: {len(all_domains_summary)}\n")
        f.write(f"Overall Average Score: {overall_avg:.4f}\n\n")
        
        # 按任务类型汇总
        f.write("Average Score by Task Type:\n")
        f.write("-" * 40 + "\n")
        task_type_scores = {}
        for domain, results in all_domains_summary.items():
            for task_name, score in results['task_scores'].items():
                if task_name not in task_type_scores:
                    task_type_scores[task_name] = []
                task_type_scores[task_name].append(score)
        
        for task_name, scores in sorted(task_type_scores.items()):
            avg_task_score = sum(scores) / len(scores) if scores else 0
            f.write(f"  {task_name:20}: {avg_task_score:.4f} (across {len(scores)} domains)\n")
    
    print(f"\n[📊] Summary report saved to {summary_file}")
    print(f"[✅] All evaluations completed! Overall Average: {overall_avg:.4f}")

if __name__ == "__main__":
    run_eval()