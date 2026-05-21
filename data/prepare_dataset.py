"""
Dataset Preparation Script for LLM Fine-Tuning
================================================
Converts CSV datasets into fine-tuning-ready formats:
  - ChatML format (Qwen, Mistral)
  - Llama3 format
  - DeepSeek format (with thinking tags)
  - Alpaca format (simplest)
  - ShareGPT format

Supports chain-of-thought, self-reflection, tool-use, and multi-step reasoning.

Usage:
    python prepare_dataset.py \
        --input data.csv \
        --output-dir ./data/processed \
        --format chatml \
        --chat-template auto \
        --train-split 0.9 \
        --add-cot true

CSV Column Detection (auto-detects or specify with --columns):
    The script tries to find these columns in your CSV:
    - instruction/user/human/question/prompt  → User message
    - response/assistant/answer/output/reply  → Assistant response
    - system/context/background               → System message (optional)
    - thinking/reasoning/explanation/chain    → Reasoning trace (optional)
    - tool_calls/tools/function_calls         → Tool usage (optional)
    - category/domain/type                    → Data category (optional)
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datasets import Dataset, DatasetDict
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# COLUMN DETECTION
# ============================================================================

COLUMN_ALIASES = {
    "instruction": ["instruction", "user", "human", "question", "prompt", "input", "query", "ask"],
    "response": ["response", "assistant", "answer", "output", "reply", "completion", "target", "solution"],
    "system": ["system", "context", "background", "preprompt", "system_prompt", "system_message"],
    "thinking": ["thinking", "reasoning", "explanation", "chain", "chain_of_thought", "rationale", "thought", "cot"],
    "tool_calls": ["tool_calls", "tools", "function_calls", "tool_use", "functions"],
    "category": ["category", "domain", "type", "topic", "label", "class"],
}


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Auto-detect CSV columns by checking column names against known aliases."""
    detected = {}
    columns_lower = {col.lower().strip(): col for col in df.columns}

    for role, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in columns_lower:
                detected[role] = columns_lower[alias]
                break

    # Validate: must have at least instruction and response
    if "instruction" not in detected:
        raise ValueError(
            f"Could not detect 'instruction' column. Available columns: {list(df.columns)}\n"
            f"Please specify with --columns instruction=YourColumnName"
        )
    if "response" not in detected:
        raise ValueError(
            f"Could not detect 'response' column. Available columns: {list(df.columns)}\n"
            f"Please specify with --columns response=YourColumnName"
        )

    logger.info(f"Detected column mapping: {detected}")
    return detected


def apply_column_overrides(detected: Dict[str, str], overrides: List[str]) -> Dict[str, str]:
    """Apply user-specified column overrides via --columns instruction=Col1 response=Col2"""
    for override in overrides:
        role, col_name = override.split("=", 1)
        role = role.strip()
        col_name = col_name.strip()
        if role in COLUMN_ALIASES:
            detected[role] = col_name
        else:
            logger.warning(f"Unknown role '{role}'. Valid roles: {list(COLUMN_ALIASES.keys())}")
    return detected


# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_text(text: Any) -> str:
    """Clean and normalize text data."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).strip()
    # Remove BOM characters
    text = text.replace("\ufeff", "")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Restore intentional newlines (for code)
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "\t")
    return text


def clean_dataframe(df: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    """Clean the DataFrame for fine-tuning quality."""
    initial_len = len(df)
    
    # Clean text columns
    for role, col in columns.items():
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Remove rows with empty instruction or response
    req_cols = [columns["instruction"], columns["response"]]
    for col in req_cols:
        df = df[df[col].str.len() > 0]

    # Remove duplicates
    df = df.drop_duplicates(subset=[columns["instruction"]], keep="first")

    # Remove extremely short responses (< 10 chars)
    df = df[df[columns["response"]].str.len() >= 10]

    # Remove extremely long samples (> 32K chars, likely noise)
    for col in req_cols:
        df = df[df[col].str.len() <= 32000]

    removed = initial_len - len(df)
    logger.info(f"Data cleaning: removed {removed} rows ({removed/initial_len*100:.1f}%), {len(df)} remaining")
    return df.reset_index(drop=True)


# ============================================================================
# FORMAT CONVERTERS
# ============================================================================

def to_chatml(row: Dict[str, str], add_cot: bool = True, add_thinking: bool = True) -> Dict:
    """
    Convert to ChatML format (used by Qwen, Mistral, etc.)
    <|im_start|>system\n{system}<|im_end|>
    <|im_start|>user\n{instruction}<|im_end|>
    <|im_start|>assistant\n{response}<|im_end|>
    """
    messages = []
    
    # System message
    system_msg = row.get("system", "")
    if not system_msg:
        system_msg = "You are an expert assistant specializing in coding and fintech. You provide accurate, well-reasoned answers with step-by-step thinking when appropriate."
    messages.append({"role": "system", "content": system_msg})
    
    # User message
    messages.append({"role": "user", "content": row["instruction"]})
    
    # Assistant response
    response_text = row["response"]
    thinking = row.get("thinking", "")
    
    if add_thinking and thinking:
        # Add reasoning trace before the response (DeepSeek-R1 style)
        response_text = f"<think/>\n{thinking}\n\n</think/>\n\n{response_text}"
    elif add_cot and not thinking:
        # Add CoT prompt encouragement in system message
        pass
    
    messages.append({"role": "assistant", "content": response_text})
    
    return {"messages": messages, "text": format_chatml_text(messages)}


def format_chatml_text(messages: List[Dict]) -> str:
    """Format messages into ChatML text string."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    return "\n".join(parts)


def to_alpaca(row: Dict[str, str], add_cot: bool = True, add_thinking: bool = True) -> Dict:
    """
    Convert to Alpaca format (simplest format)
    {"instruction": "...", "input": "...", "output": "..."}
    """
    thinking = row.get("thinking", "")
    output = row["response"]
    
    if add_thinking and thinking:
        output = f"Reasoning:\n{thinking}\n\nAnswer:\n{output}"
    
    return {
        "instruction": row["instruction"],
        "input": row.get("system", ""),
        "output": output,
    }


def to_sharegpt(row: Dict[str, str], add_cot: bool = True, add_thinking: bool = True) -> Dict:
    """
    Convert to ShareGPT format (conversations field)
    {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    """
    thinking = row.get("thinking", "")
    response = row["response"]
    
    if add_thinking and thinking:
        response = f"<think/>\n{thinking}\n\n</think/>\n\n{response}"
    
    conversations = []
    if row.get("system"):
        conversations.append({"from": "system", "value": row["system"]})
    conversations.append({"from": "human", "value": row["instruction"]})
    conversations.append({"from": "gpt", "value": response})
    
    return {"conversations": conversations}


def to_llama3(row: Dict[str, str], add_cot: bool = True, add_thinking: bool = True) -> Dict:
    """
    Convert to Llama3 format
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>
    """
    thinking = row.get("thinking", "")
    response = row["response"]
    
    if add_thinking and thinking:
        response = f"<think/>\n{thinking}\n\n</think/>\n\n{response}"
    
    messages = []
    system_msg = row.get("system", "")
    if not system_msg:
        system_msg = "You are an expert assistant specializing in coding and fintech."
    messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": row["instruction"]})
    messages.append({"role": "assistant", "content": response})
    
    return {"messages": messages, "text": format_llama3_text(messages)}


def format_llama3_text(messages: List[Dict]) -> str:
    """Format messages into Llama3 text string."""
    parts = ["<|begin_of_text|>"]
    for msg in messages:
        parts.append(f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>")
    return "".join(parts)


def to_deepseek(row: Dict[str, str], add_cot: bool = True, add_thinking: bool = True) -> Dict:
    """
    Convert to DeepSeek format with <think/> tags for reasoning.
    DeepSeek-R1 natively uses thinking blocks.
    """
    messages = []
    system_msg = row.get("system", "")
    if not system_msg:
        system_msg = "You are an expert assistant specializing in coding and fintech. Always think step-by-step before answering."
    messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": row["instruction"]})
    
    thinking = row.get("thinking", "")
    response = row["response"]
    
    if add_thinking:
        if thinking:
            # Use provided reasoning trace
            full_response = f"<think/>\n{thinking}\n</think/>\n\n{response}"
        else:
            # Generate thinking prompt from the response (extract reasoning if implicit)
            full_response = f"<think/>\nLet me think about this step by step.\n</think/>\n\n{response}"
    else:
        full_response = response
    
    messages.append({"role": "assistant", "content": full_response})
    return {"messages": messages, "text": format_chatml_text(messages)}


# ============================================================================
# REASONING AUGMENTATION
# ============================================================================

def augment_with_cot(df: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    """
    Augment dataset with chain-of-thought reasoning.
    If no 'thinking' column exists, create synthetic CoT prompts.
    """
    if "thinking" not in columns:
        logger.info("No thinking/reasoning column found. Adding CoT instruction to system prompt.")
        cot_instruction = (
            "\n\nWhen solving problems, always:\n"
            "1. Break the problem into steps\n"
            "2. Show your reasoning process\n"
            "3. Verify your answer\n"
            "4. Consider edge cases"
        )
        if "system" in columns:
            df[columns["system"]] = df[columns["system"]].apply(
                lambda x: str(x) + cot_instruction if pd.notna(x) and str(x).strip() else cot_instruction.strip()
            )
        # Create a thinking column with CoT scaffold
        df["_cot_scaffold"] = True
        columns["thinking"] = "_cot_scaffold"
    
    return df


def add_self_reflection_prompts(df: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    """Add self-reflection instruction to system prompt for reasoning improvement."""
    reflection_addition = (
        "\n\nAfter providing your initial answer, reflect on it:\n"
        "- Is my reasoning correct?\n"
        "- Are there any errors or assumptions I should reconsider?\n"
        "- Can I improve my answer?"
    )
    if "system" in columns:
        df[columns["system"]] = df[columns["system"]].apply(
            lambda x: str(x) + reflection_addition if pd.notna(x) else reflection_addition.strip()
        )
    return df


# ============================================================================
# SPLIT AND SAVE
# ============================================================================

def split_and_save(
    records: List[Dict],
    output_dir: str,
    format_name: str,
    train_split: float = 0.9,
    seed: int = 42,
) -> None:
    """Split into train/val/test and save as JSONL files."""
    np.random.seed(seed)
    n = len(records)
    indices = np.random.permutation(n)
    
    train_end = int(n * train_split)
    val_end = int(n * (train_split + (1 - train_split) / 2))
    
    train_records = [records[i] for i in indices[:train_end]]
    val_records = [records[i] for i in indices[train_end:val_end]]
    test_records = [records[i] for i in indices[val_end:]]
    
    format_dir = Path(output_dir) / format_name
    format_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in [("train", train_records), ("val", val_records), ("test", test_records)]:
        filepath = format_dir / f"{split_name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for record in split_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(split_data)} records to {filepath}")
    
    # Also save as HuggingFace Dataset
    try:
        ds_dict = {
            "train": Dataset.from_list(train_records),
            "validation": Dataset.from_list(val_records),
            "test": Dataset.from_list(test_records),
        }
        dataset_dict = DatasetDict(ds_dict)
        hf_path = format_dir / "hf_dataset"
        dataset_dict.save_to_disk(str(hf_path))
        logger.info(f"Saved HuggingFace Dataset to {hf_path}")
    except Exception as e:
        logger.warning(f"Could not save HuggingFace Dataset: {e}")
    
    # Save dataset stats
    stats = {
        "format": format_name,
        "total_samples": n,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "test_samples": len(test_records),
        "train_pct": train_split,
    }
    with open(format_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    return stats


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_dataset(records: List[Dict], format_name: str) -> Dict:
    """Validate the processed dataset and return quality metrics."""
    issues = []
    stats = {
        "total": len(records),
        "empty_messages": 0,
        "too_short": 0,
        "too_long": 0,
        "avg_user_len": 0,
        "avg_assistant_len": 0,
        "has_thinking": 0,
    }
    
    user_lens = []
    asst_lens = []
    
    for i, record in enumerate(records):
        if "messages" in record:
            for msg in record["messages"]:
                content = msg.get("content", "")
                if not content.strip():
                    stats["empty_messages"] += 1
                    issues.append(f"Record {i}: Empty {msg['role']} message")
                elif msg["role"] == "user":
                    user_lens.append(len(content))
                elif msg["role"] == "assistant":
                    asst_lens.append(len(content))
                    if "<think" in content:
                        stats["has_thinking"] += 1
        elif "instruction" in record:
            user_lens.append(len(record.get("instruction", "")))
            asst_lens.append(len(record.get("output", "")))
        
        # Check lengths
        total_len = sum(len(str(v)) for v in record.values())
        if total_len < 50:
            stats["too_short"] += 1
        elif total_len > 32000:
            stats["too_long"] += 1
            issues.append(f"Record {i}: Very long sample ({total_len} chars)")
    
    stats["avg_user_len"] = np.mean(user_lens) if user_lens else 0
    stats["avg_assistant_len"] = np.mean(asst_lens) if asst_lens else 0
    
    if issues:
        logger.warning(f"Found {len(issues)} issues. First 10:")
        for issue in issues[:10]:
            logger.warning(f"  - {issue}")
    
    logger.info(f"Dataset validation stats: {json.dumps(stats, indent=2)}")
    return stats


# ============================================================================
# MAIN
# ============================================================================

FORMAT_MAP = {
    "chatml": to_chatml,
    "alpaca": to_alpaca,
    "sharegpt": to_sharegpt,
    "llama3": to_llama3,
    "deepseek": to_deepseek,
}


def main():
    parser = argparse.ArgumentParser(description="Prepare CSV dataset for LLM fine-tuning")
    parser.add_argument("--input", required=True, help="Path to CSV file")
    parser.add_argument("--output-dir", default="./data/processed", help="Output directory")
    parser.add_argument("--format", nargs="+", default=["chatml", "alpaca", "sharegpt", "llama3", "deepseek"],
                        choices=list(FORMAT_MAP.keys()), help="Output format(s)")
    parser.add_argument("--columns", nargs="*", default=[],
                        help="Column overrides: instruction=Col1 response=Col2 system=Col3")
    parser.add_argument("--train-split", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--add-cot", type=str, default="true", help="Add chain-of-thought (true/false)")
    parser.add_argument("--add-thinking", type=str, default="true", help="Add thinking tags (true/false)")
    parser.add_argument("--add-reflection", type=str, default="false", help="Add self-reflection prompts (true/false)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--preview", action="store_true", help="Preview first 3 samples without saving")
    args = parser.parse_args()
    
    add_cot = args.add_cot.lower() == "true"
    add_thinking = args.add_thinking.lower() == "true"
    add_reflection = args.add_reflection.lower() == "true"
    
    # Load CSV
    logger.info(f"Loading CSV from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    # Detect columns
    columns = detect_columns(df)
    if args.columns:
        columns = apply_column_overrides(columns, args.columns)
    
    # Clean data
    df = clean_dataframe(df, columns)
    
    # Augment with reasoning
    if add_cot:
        df = augment_with_cot(df, columns)
    if add_reflection:
        df = add_self_reflection_prompts(df, columns)
    
    # Build row dictionaries
    rows = []
    for _, row in df.iterrows():
        row_dict = {}
        for role, col in columns.items():
            row_dict[role] = str(row[col]) if pd.notna(row[col]) else ""
        rows.append(row_dict)
    
    logger.info(f"Processed {len(rows)} rows")
    
    # Convert to each format
    all_stats = {}
    for fmt_name in args.format:
        logger.info(f"\n{'='*60}")
        logger.info(f"Converting to {fmt_name} format")
        logger.info(f"{'='*60}")
        
        converter = FORMAT_MAP[fmt_name]
        records = []
        for row_dict in rows:
            record = converter(row_dict, add_cot=add_cot, add_thinking=add_thinking)
            records.append(record)
        
        # Validate
        stats = validate_dataset(records, fmt_name)
        
        if args.preview:
            logger.info(f"\n--- Preview: {fmt_name} ---")
            for i, record in enumerate(records[:3]):
                logger.info(f"\nSample {i+1}:")
                logger.info(json.dumps(record, indent=2, ensure_ascii=False)[:500])
        else:
            # Split and save
            split_stats = split_and_save(records, args.output_dir, fmt_name, args.train_split, args.seed)
            all_stats[fmt_name] = {**stats, **split_stats}
    
    if not args.preview:
        # Save overall stats
        with open(Path(args.output_dir) / "preparation_stats.json", "w") as f:
            json.dump(all_stats, f, indent=2)
        logger.info(f"\nAll formats saved to {args.output_dir}")
        logger.info(f"Overall stats: {json.dumps(all_stats, indent=2)}")


if __name__ == "__main__":
    main()
