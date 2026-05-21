"""
Model Adapter Merger Script
============================
Merges LoRA adapter with base model for deployment.

Usage:
    python merge_adapter.py --base Qwen/Qwen2.5-Coder-7B-Instruct --adapter ./outputs/qwen-7b/ --output ./merged_model/
"""

import argparse
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def merge_lora(base_model: str, adapter_path: str, output_path: str, save_method: str = "merged_16bit"):
    """Merge LoRA adapter with base model."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth required. Install with: pip install unsloth")
        raise
    
    logger.info(f"Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        load_in_4bit=True,
    )
    
    logger.info(f"Loading adapter from: {adapter_path}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
    
    logger.info(f"Merging with method: {save_method}")
    if save_method == "merged_16bit":
        model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")
    elif save_method == "merged_4bit":
        model.save_pretrained_merged(output_path, tokenizer, save_method="merged_4bit_forced")
    elif save_method == "gguf":
        model.save_pretrained_gguf(output_path, tokenizer, quantization_method="q4_k_m")
    
    logger.info(f"Merged model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base", required=True, help="Base model name or path")
    parser.add_argument("--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--method", default="merged_16bit", 
                        choices=["merged_16bit", "merged_4bit", "gguf"],
                        help="Merge method")
    args = parser.parse_args()
    
    merge_lora(args.base, args.adapter, args.output, args.method)


if __name__ == "__main__":
    main()
