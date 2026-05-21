"""
LLM Fine-Tuning Pipeline Orchestrator
======================================
Main entry point that orchestrates the entire pipeline:
  1. Data preparation
  2. Model fine-tuning
  3. Evaluation
  4. Comparison & Recommendations

Usage:
    # Full pipeline for a specific model
    python run_pipeline.py --model qwen2.5-coder-7b --data raw_data.csv

    # Prepare data only
    python run_pipeline.py --data raw_data.csv --stage prepare

    # Fine-tune only (data already prepared)
    python run_pipeline.py --model qwen2.5-coder-7b --data ./data/processed/chatml --stage finetune

    # Evaluate only
    python run_pipeline.py --model ./outputs/qwen-7b/export_lora --stage evaluate

    # Compare all evaluated models
    python run_pipeline.py --stage compare --results-dir ./eval_results

    # Interactive model selection
    python run_pipeline.py --interactive

    # Quick test of entire pipeline
    python run_pipeline.py --model qwen2.5-coder-1.5b --data raw_data.csv --test-run
"""

import argparse
import json
import os
import sys
import subprocess
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ]
)
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).parent


# ============================================================================
# STAGE 1: DATA PREPARATION
# ============================================================================

def run_data_preparation(input_csv: str, output_dir: str, formats: List[str] = None,
                         column_overrides: List[str] = None, add_cot: bool = True,
                         add_thinking: bool = True) -> str:
    """Run data preparation pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA PREPARATION")
    logger.info("=" * 60)
    
    # First, validate the CSV
    validator_script = PIPELINE_DIR / "data" / "data_validator.py"
    logger.info(f"Validating dataset: {input_csv}")
    result = subprocess.run(
        [sys.executable, str(validator_script), "--input", input_csv],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        logger.error(f"Validation failed:\n{result.stderr}")
        # Continue anyway - warnings shouldn't block
    
    # Prepare dataset
    prepare_script = PIPELINE_DIR / "data" / "prepare_dataset.py"
    
    if formats is None:
        formats = ["chatml", "alpaca", "sharegpt", "llama3", "deepseek"]
    
    cmd = [
        sys.executable, str(prepare_script),
        "--input", input_csv,
        "--output-dir", output_dir,
        "--format", *formats,
        "--train-split", "0.9",
        "--add-cot", "true" if add_cot else "false",
        "--add-thinking", "true" if add_thinking else "false",
    ]
    
    if column_overrides:
        cmd.extend(["--columns", *column_overrides])
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        logger.error(f"Data preparation failed:\n{result.stderr}")
        raise RuntimeError("Data preparation failed")
    
    logger.info(f"Data preparation complete. Output: {output_dir}")
    return output_dir


# ============================================================================
# STAGE 2: FINE-TUNING
# ============================================================================

def run_finetuning(model_name: str, data_dir: str, output_dir: str,
                   training_config: str = "qlora_4bit", epochs: int = None,
                   lr: float = None, rank: int = None, test_run: bool = False,
                   export_formats: List[str] = None) -> str:
    """Run fine-tuning pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 2: FINE-TUNING")
    logger.info("=" * 60)
    
    finetune_script = PIPELINE_DIR / "training" / "finetune.py"
    
    cmd = [
        sys.executable, str(finetune_script),
        "--model", model_name,
        "--data", data_dir,
        "--output-dir", output_dir,
        "--training-config", training_config,
    ]
    
    if epochs:
        cmd.extend(["--epochs", str(epochs)])
    if lr:
        cmd.extend(["--lr", str(lr)])
    if rank:
        cmd.extend(["--rank", str(rank)])
    if test_run:
        cmd.append("--test-run")
    if export_formats:
        cmd.extend(["--export", *export_formats])
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        logger.error("Fine-tuning failed!")
        raise RuntimeError("Fine-tuning failed")
    
    logger.info("Fine-tuning complete!")
    return output_dir


# ============================================================================
# STAGE 3: EVALUATION
# ============================================================================

def run_evaluation(model_path: str, benchmarks: List[str] = None,
                   baseline: str = None, output_dir: str = "./eval_results",
                   quick: bool = False) -> str:
    """Run evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("STAGE 3: EVALUATION")
    logger.info("=" * 60)
    
    eval_script = PIPELINE_DIR / "evaluation" / "evaluate.py"
    
    if benchmarks is None:
        benchmarks = ["all"]
    
    cmd = [
        sys.executable, str(eval_script),
        "--model", model_path,
        "--benchmarks", *benchmarks,
        "--output", output_dir,
    ]
    
    if baseline:
        cmd.extend(["--baseline", baseline])
    if quick:
        cmd.append("--quick")
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        logger.error("Evaluation failed!")
        raise RuntimeError("Evaluation failed")
    
    logger.info("Evaluation complete!")
    return output_dir


# ============================================================================
# STAGE 4: COMPARISON & RECOMMENDATIONS
# ============================================================================

def run_comparison(results_dir: str, priority: str = "reasoning",
                   output: str = "./model_comparison.json") -> str:
    """Run model comparison and generate recommendations."""
    logger.info("=" * 60)
    logger.info("STAGE 4: COMPARISON & RECOMMENDATIONS")
    logger.info("=" * 60)
    
    compare_script = PIPELINE_DIR / "compare_models.py"
    
    cmd = [
        sys.executable, str(compare_script),
        "--results-dir", results_dir,
        "--output", output,
        "--priority", priority,
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        logger.error("Comparison failed!")
        raise RuntimeError("Comparison failed")
    
    logger.info("Comparison complete!")
    return output


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode():
    """Interactive model selection and pipeline execution."""
    print("\n" + "=" * 70)
    print("🚀 LLM Fine-Tuning Pipeline - Interactive Mode")
    print("=" * 70)
    
    # List available models
    from training.finetune import list_available_models
    models = list_available_models()
    
    print("\n📦 Available Models:")
    print("-" * 80)
    for i, m in enumerate(models):
        profile = MODEL_PROFILES.get(m["short_name"], {})
        vram = profile.get("vram_training_gb", "Unknown")
        tier = profile.get("reasoning_tier", "Unknown")
        print(f"  [{i+1}] {m['name']:<45} ({m['size']}B) | VRAM: {vram} | Reasoning: {tier}")
    
    print("\n💡 Recommendations for Coding + Fintech + Reasoning:")
    print("  🥇 DeepSeek-R1-Distill-Qwen-7B  - Best reasoning/thinking capabilities")
    print("  🥈 Qwen2.5-Coder-7B-Instruct    - Best coding + balanced reasoning")
    print("  🥉 Qwen2.5-Coder-14B-Instruct   - Best quality (tight VRAM)")
    
    # Get user selection
    choice = input("\nSelect model number (or type model short name): ").strip()
    
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            selected = models[idx]["short_name"]
        else:
            print("Invalid selection!")
            return
    else:
        selected = choice
    
    print(f"\n✅ Selected: {selected}")
    
    # Get data path
    data_path = input("Enter path to your CSV dataset: ").strip()
    if not Path(data_path).exists():
        print(f"❌ File not found: {data_path}")
        return
    
    # Ask about column mapping
    print("\n📋 Column Mapping (press Enter to auto-detect):")
    instruction_col = input("  Instruction/User column name: ").strip() or None
    response_col = input("  Response/Assistant column name: ").strip() or None
    system_col = input("  System prompt column name (optional): ").strip() or None
    thinking_col = input("  Thinking/Reasoning column name (optional): ").strip() or None
    
    column_overrides = []
    if instruction_col:
        column_overrides.append(f"instruction={instruction_col}")
    if response_col:
        column_overrides.append(f"response={response_col}")
    if system_col:
        column_overrides.append(f"system={system_col}")
    if thinking_col:
        column_overrides.append(f"thinking={thinking_col}")
    
    # Training options
    print("\n⚙️ Training Options:")
    epochs = input("  Number of epochs (default: 3): ").strip()
    lr = input("  Learning rate (default: model-specific): ").strip()
    rank = input("  LoRA rank (default: model-specific): ").strip()
    test_run = input("  Test run? (y/n, default: n): ").strip().lower() == "y"
    
    # Run pipeline
    print("\n🚀 Starting pipeline...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_dir = f"./data/processed_{timestamp}"
    output_dir = f"./outputs/{selected}_{timestamp}"
    
    # Stage 1: Prepare data
    run_data_preparation(
        input_csv=data_path,
        output_dir=processed_dir,
        column_overrides=column_overrides if column_overrides else None,
    )
    
    # Stage 2: Fine-tune
    run_finetuning(
        model_name=selected,
        data_dir=f"{processed_dir}/chatml",
        output_dir=output_dir,
        epochs=int(epochs) if epochs else None,
        lr=float(lr) if lr else None,
        rank=int(rank) if rank else None,
        test_run=test_run,
    )
    
    print("\n✅ Pipeline complete!")
    print(f"📁 Output: {output_dir}")


# Model profiles for interactive mode
MODEL_PROFILES = {
    "qwen2.5-coder-1.5b": {"vram_training_gb": "6-8", "reasoning_tier": "basic"},
    "phi-2": {"vram_training_gb": "8-10", "reasoning_tier": "moderate"},
    "qwen2.5-coder-7b": {"vram_training_gb": "16-18", "reasoning_tier": "strong"},
    "llama-3.1-8b": {"vram_training_gb": "18-20", "reasoning_tier": "strong"},
    "deepseek-coder-7b": {"vram_training_gb": "16-18", "reasoning_tier": "strong"},
    "deepseek-r1-qwen-7b": {"vram_training_gb": "16-20", "reasoning_tier": "exceptional"},
    "mistral-7b": {"vram_training_gb": "16-18", "reasoning_tier": "strong"},
    "qwen2.5-coder-14b": {"vram_training_gb": "24-28", "reasoning_tier": "very strong"},
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning Pipeline Orchestrator")
    parser.add_argument("--model", default=None, help="Model short name for fine-tuning")
    parser.add_argument("--data", required=True, help="Path to CSV data or processed data directory")
    parser.add_argument("--stage", default="all", 
                        choices=["prepare", "finetune", "evaluate", "compare", "all"],
                        help="Pipeline stage to run")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")
    parser.add_argument("--eval-results-dir", default="./eval_results", help="Evaluation results directory")
    parser.add_argument("--benchmarks", nargs="+", default=["all"],
                        choices=["perplexity", "mmlu", "humaneval", "gsm8k", "mt_bench", "fintech", "all"])
    parser.add_argument("--training-config", default="qlora_4bit", help="Training configuration")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--priority", default="reasoning", 
                        choices=["reasoning", "coding", "fintech", "balanced"])
    parser.add_argument("--columns", nargs="*", default=[], help="Column overrides")
    parser.add_argument("--test-run", action="store_true", help="Quick test run")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--baseline", default=None, help="Baseline model for comparison")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.stage in ["prepare", "all"]:
        # Determine if input is CSV or already processed
        input_path = Path(args.data)
        if input_path.suffix == ".csv":
            processed_dir = f"./data/processed_{timestamp}"
            run_data_preparation(
                input_csv=args.data,
                output_dir=processed_dir,
                column_overrides=args.columns if args.columns else None,
            )
            data_for_training = f"{processed_dir}/chatml"
        else:
            data_for_training = args.data
            logger.info(f"Using pre-processed data: {data_for_training}")
    
    if args.stage in ["finetune", "all"]:
        if not args.model:
            logger.error("--model is required for fine-tuning stage")
            sys.exit(1)
        
        data_for_training = data_for_training if args.stage == "all" else args.data
        output_dir = f"{args.output_dir}/{args.model}_{timestamp}"
        
        run_finetuning(
            model_name=args.model,
            data_dir=data_for_training,
            output_dir=output_dir,
            training_config=args.training_config,
            epochs=args.epochs,
            lr=args.lr,
            rank=args.rank,
            test_run=args.test_run,
        )
    
    if args.stage in ["evaluate", "all"]:
        # Find the model to evaluate
        if args.stage == "all" and args.model:
            model_path = f"{output_dir}/export_lora"
        elif args.model:
            model_path = args.model
        else:
            logger.error("--model is required for evaluation stage")
            sys.exit(1)
        
        run_evaluation(
            model_path=model_path,
            benchmarks=args.benchmarks,
            baseline=args.baseline,
            output_dir=args.eval_results_dir,
            quick=args.test_run,
        )
    
    if args.stage in ["compare", "all"]:
        run_comparison(
            results_dir=args.eval_results_dir,
            priority=args.priority,
            output=f"./model_comparison_{timestamp}.json",
        )
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
