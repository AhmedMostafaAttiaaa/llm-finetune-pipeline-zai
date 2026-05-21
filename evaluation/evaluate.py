"""
LLM Evaluation Pipeline
========================
Comprehensive evaluation: benchmarks, perplexity, GPT-as-judge, and custom fintech eval.

Usage:
    # Full evaluation of a fine-tuned model
    python evaluate.py --model ./outputs/qwen-7b/export_merged_16bit --benchmarks all

    # Evaluate specific benchmarks
    python evaluate.py --model ./outputs/qwen-7b/export_lora --benchmarks mmlu humaneval gsm8k

    # Quick evaluation (fewer shots, smaller subsets)
    python evaluate.py --model ./outputs/qwen-7b/export_lora --quick

    # Compare base vs fine-tuned
    python evaluate.py --model ./outputs/qwen-7b/export_lora --baseline Qwen/Qwen2.5-Coder-7B-Instruct
"""

import argparse
import json
import os
import sys
import time
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent.parent / "configs" / "evaluation"


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_for_eval(model_path: str, load_in_4bit: bool = True):
    """Load a model for evaluation."""
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            load_in_4bit=load_in_4bit,
            max_seq_length=4096,
        )
        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.warning("Unsloth not available, using standard transformers loading")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 1024, 
                      temperature: float = 0.0) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ============================================================================
# PERPLEXITY EVALUATION
# ============================================================================

def evaluate_perplexity(model, tokenizer, dataset_name: str = "wikitext-2-raw-v1", 
                        batch_size: int = 8) -> Dict:
    """Evaluate perplexity on a dataset."""
    from datasets import load_dataset
    import math
    
    logger.info(f"Evaluating perplexity on {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, split="test")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        return {"perplexity": None, "error": str(e)}
    
    # Tokenize
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    # Calculate perplexity using sliding window
    nlls = []
    max_length = 2048
    stride = 512
    
    for i in range(0, encodings.input_ids.size(1) - max_length, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - max(begin_loc, i)
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nll = outputs.loss * trg_len
        
        nlls.append(nll.item())
    
    ppl = math.exp(sum(nlls) / sum(stride for _ in nlls))
    logger.info(f"Perplexity on {dataset_name}: {ppl:.2f}")
    
    return {"perplexity": round(ppl, 2), "dataset": dataset_name}


# ============================================================================
# MMLU EVALUATION
# ============================================================================

def evaluate_mmlu(model, tokenizer, num_fewshot: int = 5, subjects: List[str] = None,
                  batch_size: int = 8) -> Dict:
    """Evaluate on MMLU benchmark using lm-eval."""
    logger.info("Evaluating on MMLU benchmark")
    
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        
        task_list = ["mmlu"]
        if subjects:
            task_list = [f"mmlu_{s}" for s in subjects]
        
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=task_list,
            num_fewshot=num_fewshot,
        )
        
        # Extract key metrics
        output = {
            "benchmark": "MMLU",
            "num_fewshot": num_fewshot,
        }
        
        if "results" in results:
            for task_name, task_results in results["results"].items():
                acc = task_results.get("acc_norm", task_results.get("acc", None))
                if acc is not None:
                    output[task_name] = round(acc * 100, 2)
        
        # Extract finance-specific if available
        if "mmlu_finance" in output or "mmlu_accounting" in output or "mmlu_econometrics" in output:
            finance_scores = [v for k, v in output.items() if "financ" in k.lower() or "account" in k.lower() or "econom" in k.lower()]
            if finance_scores:
                output["finance_avg"] = round(sum(finance_scores) / len(finance_scores), 2)
        
        logger.info(f"MMLU Results: {output}")
        return output
        
    except ImportError:
        logger.warning("lm-eval not installed. Install with: pip install lm-eval")
        return {"benchmark": "MMLU", "error": "lm-eval not installed"}


# ============================================================================
# HUMANEVAL EVALUATION
# ============================================================================

def evaluate_humaneval(model, tokenizer, num_samples: int = 1, batch_size: int = 8) -> Dict:
    """Evaluate on HumanEval code generation benchmark."""
    logger.info("Evaluating on HumanEval")
    
    try:
        from datasets import load_dataset
        from evaluate import load as load_metric
        
        dataset = load_dataset("openai/openai_humaneval", split="test")
        
        results = []
        for i, problem in enumerate(dataset):
            prompt = problem["prompt"]
            test = problem["test"]
            task_id = problem["task_id"]
            
            # Format prompt for code generation
            code_prompt = f"Complete the following Python function:\n\n{prompt}"
            
            # Generate
            response = generate_response(model, tokenizer, code_prompt, max_new_tokens=512, temperature=0.0)
            
            # Extract code from response
            completion = extract_code(response)
            
            results.append({
                "task_id": task_id,
                "completion": completion,
                "prompt": prompt,
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"HumanEval: {i+1}/{len(dataset)} problems evaluated")
        
        # Save completions for external evaluation
        completions_path = Path("./eval_results/humaneval_completions.jsonl")
        completions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(completions_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        
        # Run evaluation using exec-based checking (simple)
        pass_count = 0
        for r in results:
            try:
                # Combine prompt + completion and test
                full_code = r["prompt"] + r["completion"]
                exec_globals = {}
                exec(full_code, exec_globals)
                exec(r["prompt"].split("def ")[1].split("(")[0] + "()", exec_globals)  # Basic syntax check
                pass_count += 1
            except Exception:
                pass
        
        pass_rate = pass_count / len(results) * 100
        
        output = {
            "benchmark": "HumanEval",
            "pass_rate_syntax": round(pass_rate, 2),
            "num_problems": len(results),
            "note": "For accurate pass@1, run: python -m evaluate --humaneval ./eval_results/humaneval_completions.jsonl",
        }
        
        logger.info(f"HumanEval Results: {output}")
        return output
        
    except Exception as e:
        logger.error(f"HumanEval evaluation failed: {e}")
        return {"benchmark": "HumanEval", "error": str(e)}


def extract_code(text: str) -> str:
    """Extract code from model response, handling various formats."""
    # Try to extract from code blocks
    import re
    
    # Check for ```python ... ``` blocks
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # Check for ``` ... ``` blocks
    code_blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # Return everything after the function signature
    lines = text.split("\n")
    code_lines = []
    for line in lines:
        if line.strip().startswith("def ") or line.strip().startswith("class ") or code_lines:
            code_lines.append(line)
    
    if code_lines:
        return "\n".join(code_lines)
    
    return text.strip()


# ============================================================================
# GSM8K EVALUATION
# ============================================================================

def evaluate_gsm8k(model, tokenizer, num_fewshot: int = 8, batch_size: int = 8) -> Dict:
    """Evaluate on GSM8K math reasoning benchmark."""
    logger.info("Evaluating on GSM8K")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("gsm8k", "main", split="test")
        
        correct = 0
        total = 0
        
        few_shot_examples = """Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market for $2 each. How much does she make every day?
Answer: Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * $2 = $18 every day. The answer is 18.

Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: It takes 2/2 = 1 bolt of white fiber. So the total amount of bolts is 2 + 1 = 3. The answer is 3.

Problem: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
Answer: The cost of the house and repairs came out to 80,000 + 50,000 = 130,000. The value increased by 150%, so the new value is 80,000 * 2.5 = 200,000. The profit is 200,000 - 130,000 = 70,000. The answer is 70000.
"""
        
        for i, problem in enumerate(dataset):
            question = problem["question"]
            answer = problem["answer"]
            # Extract final number from answer
            try:
                expected = float(answer.split("####")[-1].strip().replace(",", ""))
            except (ValueError, IndexError):
                continue
            
            prompt = f"{few_shot_examples}\nProblem: {question}\nAnswer:"
            response = generate_response(model, tokenizer, prompt, max_new_tokens=512)
            
            # Extract number from response
            try:
                predicted = extract_number(response)
                if predicted is not None and abs(predicted - expected) < 0.01:
                    correct += 1
            except Exception:
                pass
            
            total += 1
            
            if (i + 1) % 50 == 0:
                logger.info(f"GSM8K: {i+1}/{len(dataset)} accuracy: {correct/total*100:.1f}%")
        
        accuracy = correct / total * 100 if total > 0 else 0
        
        output = {
            "benchmark": "GSM8K",
            "accuracy": round(accuracy, 2),
            "correct": correct,
            "total": total,
            "num_fewshot": num_fewshot,
        }
        
        logger.info(f"GSM8K Results: {output}")
        return output
        
    except Exception as e:
        logger.error(f"GSM8K evaluation failed: {e}")
        return {"benchmark": "GSM8K", "error": str(e)}


def extract_number(text: str) -> Optional[float]:
    """Extract the final numerical answer from a response."""
    import re
    
    # Try to find "The answer is X" pattern
    match = re.search(r"(?:the answer is|the answer is:?\s*)([+-]?\d[\d,]*\.?\d*)", text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    
    # Try to find #### X pattern (GSM8K format)
    match = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    
    # Return last number in the text
    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    
    return None


# ============================================================================
# MT-BENCH EVALUATION
# ============================================================================

def evaluate_mt_bench(model, tokenizer, judge_model: str = "gpt-4o-mini", 
                      num_samples: int = None) -> Dict:
    """Evaluate on MT-Bench using GPT-as-judge."""
    logger.info("Evaluating on MT-Bench")
    
    try:
        from datasets import load_dataset
        
        # Load MT-Bench questions
        mt_bench_data = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        
        if num_samples:
            mt_bench_data = mt_bench_data.select(range(min(num_samples, len(mt_bench_data))))
        
        # Generate responses
        results = []
        for i, item in enumerate(mt_bench_data):
            prompt = item.get("prompt", item.get("turns", [""])[0] if "turns" in item else "")
            if not prompt:
                continue
            
            response = generate_response(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.7)
            
            results.append({
                "question_id": item.get("question_id", i),
                "category": item.get("category", "unknown"),
                "prompt": prompt,
                "response": response,
                "turn": item.get("turn", 1),
            })
        
        # Save for judge evaluation
        results_path = Path("./eval_results/mt_bench_responses.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Try GPT-as-judge
        scores = gpt_as_judge(results, judge_model) if judge_model else {}
        
        output = {
            "benchmark": "MT-Bench",
            "num_questions": len(results),
            "judge_model": judge_model,
            **scores,
        }
        
        logger.info(f"MT-Bench Results: {output}")
        return output
        
    except Exception as e:
        logger.error(f"MT-Bench evaluation failed: {e}")
        return {"benchmark": "MT-Bench", "error": str(e)}


# ============================================================================
# GPT-AS-JUDGE
# ============================================================================

def gpt_as_judge(responses: List[Dict], judge_model: str = "gpt-4o-mini") -> Dict:
    """Use GPT model to judge response quality."""
    logger.info(f"Running GPT-as-judge with {judge_model}")
    
    try:
        import openai
    except ImportError:
        logger.warning("openai not installed. Skipping GPT-as-judge. Install with: pip install openai")
        return {"judge_scores": None, "error": "openai not installed"}
    
    client = openai.OpenAI()  # Uses OPENAI_API_KEY env variable
    
    judge_prompt = """You are a judge evaluating an AI assistant's response. Rate the response on these criteria:

1. Relevance (1-10): Does the response address the question?
2. Accuracy (1-10): Is the information correct?
3. Completeness (1-10): Does it cover all aspects?
4. Reasoning Quality (1-10): Is the reasoning sound and well-structured?
5. Code Correctness (1-10, if applicable): Is any code correct and functional?

Question: {question}
Response: {response}

Provide your ratings as a JSON object with keys: relevance, accuracy, completeness, reasoning_quality, code_correctness (use N/A if not applicable), and an overall_score (1-10). Also include brief feedback."""
    
    all_scores = defaultdict(list)
    
    for i, item in enumerate(responses):
        try:
            prompt = judge_prompt.format(question=item["prompt"], response=item["response"])
            
            completion = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            
            scores_text = completion.choices[0].message.content
            scores = json.loads(scores_text)
            
            for key, value in scores.items():
                if key != "feedback" and isinstance(value, (int, float)):
                    all_scores[key].append(value)
            
        except Exception as e:
            logger.warning(f"Judge failed for item {i}: {e}")
            continue
    
    # Aggregate scores
    avg_scores = {}
    for key, values in all_scores.items():
        if values:
            avg_scores[key] = round(sum(values) / len(values), 2)
    
    logger.info(f"GPT-Judge Average Scores: {avg_scores}")
    return {"judge_scores": avg_scores}


# ============================================================================
# CUSTOM FINTECH EVALUATION
# ============================================================================

def evaluate_fintech(model, tokenizer, custom_prompts_file: str = None) -> Dict:
    """Evaluate on custom fintech-specific prompts."""
    logger.info("Evaluating on fintech-specific tasks")
    
    # Default fintech evaluation prompts
    fintech_prompts = {
        "regulatory_compliance": [
            "Explain the key requirements of PSD2 for payment service providers in the EU.",
            "What are the main compliance requirements of SOX for fintech companies?",
        ],
        "risk_assessment": [
            "Describe a credit risk assessment framework for a peer-to-peer lending platform.",
            "How would you implement a real-time fraud detection system for online payments?",
        ],
        "financial_analysis": [
            "Calculate the Sharpe ratio for a portfolio with 12% return, 3% risk-free rate, and 8% standard deviation.",
            "Explain how to perform a discounted cash flow (DCF) valuation for a fintech startup.",
        ],
        "algorithmic_trading": [
            "Write a Python function implementing a simple moving average crossover trading strategy.",
            "Explain the concept of mean reversion in algorithmic trading with a code example.",
        ],
        "fraud_detection": [
            "Design a feature engineering pipeline for credit card fraud detection using transaction data.",
            "Write Python code to implement an anomaly detection system for financial transactions using isolation forests.",
        ],
        "portfolio_optimization": [
            "Implement Markowitz portfolio optimization in Python using scipy.optimize.",
            "Explain the Capital Asset Pricing Model (CAPM) and write code to calculate expected returns.",
        ],
    }
    
    # Load custom prompts if provided
    if custom_prompts_file and Path(custom_prompts_file).exists():
        with open(custom_prompts_file) as f:
            fintech_prompts = json.load(f)
    
    results = {}
    all_responses = []
    
    for category, prompts in fintech_prompts.items():
        cat_results = []
        for prompt in prompts:
            response = generate_response(model, tokenizer, prompt, max_new_tokens=1024)
            cat_results.append({
                "category": category,
                "prompt": prompt,
                "response": response,
            })
            all_responses.append(cat_results[-1])
        
        results[category] = {
            "num_prompts": len(prompts),
            "sample_response": cat_results[0]["response"][:200] + "..." if cat_results else "",
        }
    
    # Save results
    results_path = Path("./eval_results/fintech_eval.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)
    
    output = {
        "benchmark": "FinTech Custom",
        "categories": list(fintech_prompts.keys()),
        "total_prompts": sum(len(v) for v in fintech_prompts.values()),
        "results_path": str(results_path),
        "category_summaries": results,
    }
    
    logger.info(f"FinTech Eval Results saved to {results_path}")
    return output


# ============================================================================
# MAIN EVALUATION RUNNER
# ============================================================================

def run_evaluation(model, tokenizer, benchmarks: List[str], config: Dict, 
                   baseline_model=None, baseline_tokenizer=None) -> Dict:
    """Run all specified benchmark evaluations."""
    
    all_results = {"timestamp": datetime.now().isoformat(), "benchmarks": {}}
    
    benchmark_runners = {
        "perplexity": lambda: evaluate_perplexity(model, tokenizer),
        "mmlu": lambda: evaluate_mmlu(model, tokenizer),
        "humaneval": lambda: evaluate_humaneval(model, tokenizer),
        "gsm8k": lambda: evaluate_gsm8k(model, tokenizer),
        "mt_bench": lambda: evaluate_mt_bench(model, tokenizer, 
                                               judge_model=config.get("gpt_judge", {}).get("judge_model", "gpt-4o-mini")),
        "fintech": lambda: evaluate_fintech(model, tokenizer),
    }
    
    if "all" in benchmarks:
        benchmarks = list(benchmark_runners.keys())
    
    for bench_name in benchmarks:
        if bench_name in benchmark_runners:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {bench_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = benchmark_runners[bench_name]()
                all_results["benchmarks"][bench_name] = result
            except Exception as e:
                logger.error(f"Benchmark {bench_name} failed: {e}")
                all_results["benchmarks"][bench_name] = {"error": str(e)}
        else:
            logger.warning(f"Unknown benchmark: {bench_name}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLM")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--benchmarks", nargs="+", default=["all"],
                        choices=["perplexity", "mmlu", "humaneval", "gsm8k", "mt_bench", "fintech", "all"],
                        help="Benchmarks to run")
    parser.add_argument("--baseline", default=None, help="Baseline model for comparison")
    parser.add_argument("--config", default=str(EVAL_DIR / "benchmarks.yaml"), help="Eval config file")
    parser.add_argument("--output", default="./eval_results", help="Results output directory")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation mode")
    parser.add_argument("--load-4bit", action="store_true", default=True, help="Load model in 4-bit")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model_for_eval(args.model, load_in_4bit=args.load_4bit)
    
    # Load baseline if specified
    baseline_model, baseline_tokenizer = None, None
    if args.baseline:
        logger.info(f"Loading baseline model: {args.baseline}")
        baseline_model, baseline_tokenizer = load_model_for_eval(args.baseline, load_in_4bit=True)
    
    # Run evaluation
    results = run_evaluation(model, tokenizer, args.benchmarks, config, baseline_model, baseline_tokenizer)
    
    # Run baseline evaluation for comparison
    if baseline_model:
        logger.info("\n" + "="*60)
        logger.info("Evaluating BASELINE model for comparison")
        logger.info("="*60)
        baseline_results = run_evaluation(baseline_model, baseline_tokenizer, args.benchmarks, config)
        results["baseline"] = baseline_results
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"eval_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    for bench_name, bench_result in results.get("benchmarks", {}).items():
        print(f"\n{bench_name.upper()}:")
        if "error" in bench_result:
            print(f"  ❌ Error: {bench_result['error']}")
        else:
            for key, value in bench_result.items():
                if key not in ["note", "results_path"] and not isinstance(value, dict):
                    print(f"  {key}: {value}")
    
    print(f"\nFull results saved to: {results_file}")
    
    if baseline_model:
        print("\nBASELINE COMPARISON:")
        for bench_name in results.get("benchmarks", {}):
            ft = results["benchmarks"].get(bench_name, {})
            bl = results.get("baseline", {}).get("benchmarks", {}).get(bench_name, {})
            if "error" not in ft and "error" not in bl:
                print(f"  {bench_name}: Fine-tuned vs Baseline comparison available in results file")


if __name__ == "__main__":
    main()
