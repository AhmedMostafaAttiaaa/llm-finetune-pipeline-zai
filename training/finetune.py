"""
LLM Fine-Tuning Script with Unsloth + LoRA/QLoRA
==================================================
Supports all configured models with automatic setup.

Usage:
    # Fine-tune a specific model
    python finetune.py --model qwen2.5-coder-7b --data ./data/processed/chatml

    # Fine-tune with custom parameters
    python finetune.py --model llama-3.1-8b --data ./data/processed/chatml \
        --epochs 5 --lr 1e-4 --rank 32 --max-seq-length 2048

    # List available models
    python finetune.py --list-models

    # Quick test run (1 epoch, small subset)
    python finetune.py --model qwen2.5-coder-1.5b --data ./data/processed/chatml --test-run
"""

import argparse
import json
import os
import sys
import yaml
import torch
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
MODELS_DIR = CONFIGS_DIR / "models"


# ============================================================================
# CONFIG LOADING
# ============================================================================

def list_available_models() -> list:
    """List all available model configurations."""
    models = []
    for f in sorted(MODELS_DIR.glob("*.yaml")):
        with open(f) as fh:
            config = yaml.safe_load(fh)
        models.append({
            "file": f.stem,
            "name": config["model"]["name"],
            "short_name": config["model"]["short_name"],
            "size": config["model"]["size_billions"],
            "family": config["model"]["family"],
            "type": config["model"]["type"],
            "context_length": config["model"]["context_length"],
        })
    return models


def load_model_config(model_name: str) -> Dict:
    """Load model configuration by short name or YAML filename."""
    # Try exact filename match
    filepath = MODELS_DIR / f"{model_name}.yaml"
    if not filepath.exists():
        # Try partial match
        for f in MODELS_DIR.glob("*.yaml"):
            if model_name in f.stem:
                filepath = f
                break
        else:
            available = [f.stem for f in MODELS_DIR.glob("*.yaml")]
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    
    with open(filepath) as fh:
        config = yaml.safe_load(fh)
    return config


def load_training_config(config_name: str = "qlora_4bit") -> Dict:
    """Load training configuration."""
    filepath = CONFIGS_DIR / "training" / f"{config_name}.yaml"
    with open(filepath) as fh:
        return yaml.safe_load(fh)


# ============================================================================
# UNSLOTH SETUP & TRAINING
# ============================================================================

def setup_unsloth(model_config: Dict, training_config: Dict, max_seq_length: int = 4096):
    """Initialize model and tokenizer with Unsloth optimizations."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth not installed. Install with: pip install unsloth")
        raise
    
    model_name = model_config["model"]["name"]
    load_in_4bit = training_config["quantization"]["load_in_4bit"]
    
    logger.info(f"Loading model: {model_name}")
    logger.info(f"4-bit quantization: {load_in_4bit}")
    logger.info(f"Max sequence length: {max_seq_length}")
    
    dtype = None  # Auto-detect (Ampere+ → bfloat16, older → float16)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        trust_remote_code=model_config["model"].get("trust_remote_code", False),
    )
    
    logger.info(f"Model loaded. Vocab size: {tokenizer.vocab_size}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    return model, tokenizer


def apply_lora(model, model_config: Dict, lora_overrides: Dict = None):
    """Apply LoRA adapter to the model."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise
    
    lora_config = model_config["lora"].copy()
    if lora_overrides:
        lora_config.update(lora_overrides)
    
    logger.info(f"Applying LoRA: rank={lora_config['rank']}, alpha={lora_config['alpha']}")
    logger.info(f"Target modules: {lora_config['target_modules']}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["rank"],
        target_modules=lora_config["target_modules"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
        random_state=42,
        use_rslora=False,  # Rank-stabilized LoRA
        loftq_config=None,
    )
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model


def load_dataset(data_dir: str, tokenizer, max_seq_length: int, model_config: Dict):
    """Load and format the fine-tuning dataset."""
    from datasets import load_from_disk, load_dataset as hf_load_dataset
    from unsloth.chat_templates import get_chat_template
    
    data_path = Path(data_dir)
    
    # Try loading HuggingFace dataset format first
    hf_path = data_path / "hf_dataset"
    if hf_path.exists():
        logger.info(f"Loading HuggingFace dataset from {hf_path}")
        dataset = load_from_disk(str(hf_path))
    else:
        # Load JSONL files
        train_file = data_path / "train.jsonl"
        if not train_file.exists():
            raise FileNotFoundError(f"No training data found in {data_dir}")
        
        logger.info(f"Loading JSONL dataset from {data_dir}")
        dataset = hf_load_dataset("json", data_files={
            "train": str(data_path / "train.jsonl"),
            "validation": str(data_path / "val.jsonl") if (data_path / "val.jsonl").exists() else None,
        })
    
    # Apply chat template based on model family
    chat_template = model_config["model"].get("chat_template", "chatml")
    logger.info(f"Using chat template: {chat_template}")
    
    # Formatting function for training
    def formatting_prompts_func(examples):
        convos = examples.get("messages", examples.get("conversations", None))
        if convos is None:
            return {"text": [""] * len(examples[list(examples.keys())[0]])}
        
        texts = []
        for convo in convos:
            # Use tokenizer's chat template if available
            try:
                text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            except Exception:
                # Fallback: manual formatting
                text = format_conversation(convo, chat_template)
            texts.append(text)
        return {"text": texts}
    
    # Apply formatting
    if "messages" in dataset["train"].column_names or "conversations" in dataset["train"].column_names:
        dataset = dataset.map(formatting_prompts_func, batched=True)
    elif "text" not in dataset["train"].column_names:
        raise ValueError("Dataset must have 'messages', 'conversations', or 'text' column")
    
    logger.info(f"Dataset: {len(dataset['train'])} train, {len(dataset.get('validation', []))} val")
    return dataset


def format_conversation(messages: list, template: str) -> str:
    """Manually format conversation when tokenizer chat template is unavailable."""
    if template == "chatml":
        parts = []
        for msg in messages:
            role = msg["role"] if "role" in msg else msg.get("from", "user")
            content = msg.get("content", msg.get("value", ""))
            if role == "system" or role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user" or role == "human":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant" or role == "gpt":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        return "\n".join(parts)
    elif template == "llama3":
        parts = ["<|begin_of_text|>"]
        for msg in messages:
            role = msg.get("role", msg.get("from", "user"))
            content = msg.get("content", msg.get("value", ""))
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        return "".join(parts)
    else:
        # Default: just concatenate
        return "\n".join(msg.get("content", msg.get("value", "")) for msg in messages)


def train(model, tokenizer, dataset, model_config: Dict, training_config: Dict, args):
    """Run the training loop."""
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainingArguments
    
    # Merge training configs
    train_params = model_config["training"].copy()
    
    # Apply CLI overrides
    if args.epochs:
        train_params["num_train_epochs"] = args.epochs
    if args.lr:
        train_params["learning_rate"] = args.lr
    if args.max_seq_length:
        train_params["max_seq_length"] = args.max_seq_length
    if args.batch_size:
        train_params["per_device_train_batch_size"] = args.batch_size
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{model_config['model']['short_name']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test run: limit data
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation", None)
    
    if args.test_run:
        train_dataset = train_dataset.select(range(min(100, len(train_dataset))))
        if eval_dataset:
            eval_dataset = eval_dataset.select(range(min(20, len(eval_dataset))))
        train_params["num_train_epochs"] = 1
        train_params["max_steps"] = 50
        logger.info("TEST RUN: Using subset of data, 1 epoch max")
    
    # Memory config
    mem_config = training_config["memory"]
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_params["per_device_train_batch_size"],
        gradient_accumulation_steps=train_params["gradient_accumulation_steps"],
        warmup_ratio=train_params["warmup_ratio"],
        num_train_epochs=train_params["num_train_epochs"],
        learning_rate=train_params["learning_rate"],
        fp16=mem_config.get("fp16", False),
        bf16=mem_config.get("bf16", True),
        logging_steps=training_config["logging"]["logging_steps"],
        save_strategy=training_config["logging"]["save_strategy"],
        save_steps=training_config["logging"]["save_steps"],
        save_total_limit=training_config["logging"]["save_total_limit"],
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=training_config["logging"]["eval_steps"] if eval_dataset else None,
        optim=train_params["optim"],
        weight_decay=train_params["weight_decay"],
        max_grad_norm=train_params["max_grad_norm"],
        lr_scheduler_type=train_params["lr_scheduler_type"],
        gradient_checkpointing=mem_config["gradient_checkpointing"],
        gradient_checkpointing_kwargs=mem_config.get("gradient_checkpointing_kwargs", {}),
        report_to=training_config["logging"]["report_to"],
        seed=42,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    
    # Override max_steps for test run
    if args.test_run and "max_steps" in train_params:
        training_args.max_steps = train_params["max_steps"]
    
    logger.info(f"Training arguments: {training_args}")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=train_params["max_seq_length"],
        packing=True,  # Unsloth packing for efficiency
        args=training_args,
    )
    
    # Log GPU stats before training
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        gpu_mem = torch.cuda.get_device_name(0)
        logger.info(f"GPU: {gpu_mem}")
        logger.info(f"GPU Memory: {gpu_stats.total_mem / 1024**3:.1f} GB")
        logger.info(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
        logger.info(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    
    # Train!
    logger.info("Starting training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/3600:.2f} hours")
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    
    # Save training stats
    stats = {
        "model": model_config["model"]["name"],
        "training_time_hours": training_time / 3600,
        "train_loss": train_result.training_loss,
        "train_samples": len(train_dataset),
        "epochs": train_params["num_train_epochs"],
        "learning_rate": train_params["learning_rate"],
        "lora_rank": model_config["lora"]["rank"],
        "max_seq_length": train_params["max_seq_length"],
        "quantization": "4-bit QLoRA" if training_config["quantization"]["load_in_4bit"] else "LoRA FP16",
        "output_dir": str(output_dir),
        "timestamp": timestamp,
    }
    
    stats_path = output_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Training stats saved to {stats_path}")
    
    return trainer, stats


def save_inference_model(trainer, model_config: Dict, output_dir: str, methods: list = None):
    """Save model in various formats for deployment."""
    if methods is None:
        methods = ["lora", "merged_16bit", "gguf"]
    
    base_dir = Path(output_dir)
    
    for method in methods:
        save_path = base_dir / f"export_{method}"
        logger.info(f"Saving model as {method} to {save_path}")
        
        try:
            if method == "lora":
                trainer.model.save_pretrained(str(save_path))
                trainer.tokenizer.save_pretrained(str(save_path))
            elif method == "merged_16bit":
                trainer.model.save_pretrained_merged(str(save_path), trainer.tokenizer, save_method="merged_16bit")
            elif method == "merged_4bit":
                trainer.model.save_pretrained_merged(str(save_path), trainer.tokenizer, save_method="merged_4bit_forced")
            elif method == "gguf":
                # Save as GGUF for llama.cpp / Ollama deployment
                try:
                    trainer.model.save_pretrained_gguf(str(save_path), trainer.tokenizer)
                except Exception as e:
                    logger.warning(f"GGUF export failed: {e}. Try: model.save_pretrained_gguf(path, tokenizer, quantization_method='q4_k_m')")
                    try:
                        trainer.model.save_pretrained_gguf(str(save_path), trainer.tokenizer, quantization_method="q4_k_m")
                    except Exception as e2:
                        logger.error(f"GGUF export also failed with q4_k_m: {e2}")
            elif method == "vllm":
                # Save merged 16-bit for vLLM serving
                trainer.model.save_pretrained_merged(str(save_path), trainer.tokenizer, save_method="merged_16bit")
        except Exception as e:
            logger.error(f"Failed to save as {method}: {e}")
    
    logger.info("Model export complete")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with Unsloth + LoRA/QLoRA")
    parser.add_argument("--model", required=True, help="Model short name (e.g., qwen2.5-coder-7b)")
    parser.add_argument("--data", required=True, help="Path to processed dataset directory")
    parser.add_argument("--output-dir", default="./outputs", help="Output directory")
    parser.add_argument("--training-config", default="qlora_4bit", help="Training config (qlora_4bit or lora_fp16)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--rank", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Override max sequence length")
    parser.add_argument("--export", nargs="+", default=["lora", "merged_16bit", "gguf"],
                        choices=["lora", "merged_16bit", "merged_4bit", "gguf", "vllm"],
                        help="Export formats")
    parser.add_argument("--test-run", action="store_true", help="Quick test run (1 epoch, small subset)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--skip-export", action="store_true", help="Skip model export after training")
    args = parser.parse_args()
    
    if args.list_models:
        models = list_available_models()
        print("\nAvailable Models:")
        print("-" * 80)
        print(f"{'Short Name':<30} {'HF Name':<45} {'Size':>6}")
        print("-" * 80)
        for m in models:
            print(f"{m['short_name']:<30} {m['name']:<45} {m['size']:>5.1f}B")
        return
    
    # Load configs
    model_config = load_model_config(args.model)
    training_config = load_training_config(args.training_config)
    
    logger.info(f"Model: {model_config['model']['name']}")
    logger.info(f"Training: {args.training_config}")
    
    # Setup
    max_seq_length = args.max_seq_length or model_config["training"]["max_seq_length"]
    model, tokenizer = setup_unsloth(model_config, training_config, max_seq_length)
    
    # Apply LoRA
    lora_overrides = {}
    if args.rank:
        lora_overrides["rank"] = args.rank
        lora_overrides["alpha"] = args.rank * 2  # Standard practice: alpha = 2 * rank
    model = apply_lora(model, model_config, lora_overrides)
    
    # Load dataset
    dataset = load_dataset(args.data, tokenizer, max_seq_length, model_config)
    
    # Train
    trainer, stats = train(model, tokenizer, dataset, model_config, training_config, args)
    
    # Export
    if not args.skip_export:
        output_dir = stats["output_dir"]
        save_inference_model(trainer, model_config, output_dir, args.export)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {stats['model']}")
    print(f"Training time: {stats['training_time_hours']:.2f} hours")
    print(f"Train loss: {stats['train_loss']:.4f}")
    print(f"Output: {stats['output_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
