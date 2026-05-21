"""
Generate the Interactive Jupyter Notebook for LLM Fine-Tuning Pipeline
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

cells = []

# ============================================================
# TITLE & INTRO
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""# 🚀 LLM Fine-Tuning Pipeline — Interactive Notebook

**Domain:** Coding & Fintech | **Hardware:** Single GPU (30GB VRAM) | **Method:** QLoRA/LoRA + Unsloth

This notebook walks you through the **entire pipeline** step by step:
1. **Environment Setup** — Install dependencies and verify GPU
2. **Data Exploration** — Validate and preview your CSV dataset
3. **Dataset Preparation** — Convert CSV to fine-tuning formats with reasoning augmentation
4. **Model Selection** — Compare and choose the right model for your needs
5. **Fine-Tuning** — Train with Unsloth + QLoRA/LoRA
6. **Evaluation** — Benchmark your fine-tuned model
7. **Model Comparison** — Compare results and get recommendations
8. **Export & Deploy** — Save model for production

---

**🎯 Quick Recommendation:**
| Your Priority | Best Model |
|---|---|
| Reasoning & Thinking | **DeepSeek-R1-Distill-Qwen-7B** |
| Coding + Fintech Balance | **Qwen2.5-Coder-7B-Instruct** |
| Maximum Quality | **Qwen2.5-Coder-14B-Instruct** (tight VRAM!) |
| Fast Prototyping | **Qwen2.5-Coder-1.5B-Instruct** |
"""))

# ============================================================
# STEP 0: CONFIGURATION
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""---
## ⚙️ Step 0: Configuration

Set your paths and preferences here. These variables are used throughout the notebook.
"""))

cells.append(nbf.v4.new_code_cell('''# ============================================================
# ⚙️ CONFIGURATION — Edit these values for your setup
# ============================================================

# --- Data ---
DATA_CSV = "your_dataset.csv"          # Path to your CSV file
DATA_DIR = "./data/processed"           # Output directory for processed data
TRAIN_SPLIT = 0.9                       # Train/val/test split ratio

# --- Column Mapping (auto-detect if empty) ---
# Set these if your CSV column names aren\'t auto-detected
COLUMN_INSTRUCTION = ""   # e.g., "question", "prompt", "user_input"
COLUMN_RESPONSE = ""      # e.g., "answer", "output", "assistant_reply"
COLUMN_SYSTEM = ""        # e.g., "context", "system_prompt"
COLUMN_THINKING = ""      # e.g., "reasoning", "chain_of_thought"

# --- Model Selection ---
MODEL_NAME = "deepseek-r1-qwen-7b"     # Model short name (see list below)
# Options: qwen2.5-coder-1.5b, phi-2, qwen2.5-coder-7b, llama-3.1-8b,
#          deepseek-coder-7b, deepseek-r1-qwen-7b, mistral-7b, qwen2.5-coder-14b

# --- Training ---
TRAINING_CONFIG = "qlora_4bit"          # "qlora_4bit" or "lora_fp16"
EPOCHS = 3                              # Number of training epochs
LEARNING_RATE = None                     # None = use model default
LORA_RANK = None                         # None = use model default (64 for 7B, 32 for 14B)
MAX_SEQ_LENGTH = None                    # None = use model default
BATCH_SIZE = None                        # None = use model default
OUTPUT_DIR = "./outputs"                 # Training output directory

# --- Evaluation ---
BENCHMARKS = ["mmlu", "gsm8k", "humaneval", "mt_bench", "fintech"]
BASELINE_MODEL = None                    # Compare against baseline (optional)
GPT_JUDGE = True                         # Use GPT-as-judge for MT-Bench (needs OPENAI_API_KEY)

# --- Export ---
EXPORT_FORMATS = ["lora", "merged_16bit", "gguf"]

# --- Reasoning Augmentation ---
ADD_CHAIN_OF_THOUGHT = True              # Add CoT scaffolding to system prompt
ADD_THINKING_TAGS = True                 # Wrap reasoning in <think/>...</think/> tags
ADD_SELF_REFLECTION = False              # Add self-reflection prompts

print("✅ Configuration loaded!")
print(f"   Data: {DATA_CSV}")
print(f"   Model: {MODEL_NAME}")
print(f"   Training: {TRAINING_CONFIG}, {EPOCHS} epochs")
print(f"   Reasoning: CoT={ADD_CHAIN_OF_THOUGHT}, Thinking={ADD_THINKING_TAGS}, Reflection={ADD_SELF_REFLECTION}")
'''))

# ============================================================
# STEP 1: ENVIRONMENT SETUP
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""---
## 📦 Step 1: Environment Setup

Install required packages and verify GPU availability.
"""))

cells.append(nbf.v4.new_code_cell('''# Install core dependencies (run once)
# !pip install unsloth  # Uncomment if Unsloth not installed
# !pip install -r requirements.txt  # Uncomment for full install

# Quick install of essentials
import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

for pkg in ["torch", "transformers", "peft", "trl", "datasets", "pandas", "pyyaml", "accelerate", "bitsandbytes"]:
    install_if_missing(pkg)

print("✅ Core packages installed!")
'''))

cells.append(nbf.v4.new_code_cell('''# Verify GPU
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"🎮 GPU: {gpu_name}")
    print(f"📊 VRAM: {gpu_vram:.1f} GB")
    
    if gpu_vram >= 24:
        print("✅ Sufficient VRAM for 7B+ models with QLoRA 4-bit")
    elif gpu_vram >= 16:
        print("⚠️ Limited VRAM — stick to 7B models with QLoRA 4-bit")
    elif gpu_vram >= 8:
        print("⚠️ Low VRAM — use 1.5B-3B models only")
    else:
        print("❌ Insufficient VRAM for fine-tuning")
else:
    print("❌ No GPU detected! Fine-tuning requires a CUDA-capable GPU.")

# Check CUDA version
print(f"\\nCUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")
'''))

cells.append(nbf.v4.new_code_cell('''# Check Unsloth availability
try:
    from unsloth import FastLanguageModel
    print("✅ Unsloth is installed — 2x faster training enabled!")
except ImportError:
    print("⚠️ Unsloth not installed. Install with: pip install unsloth")
    print("   Training will still work but will be slower without Unsloth optimizations.")

# Check optional packages
optional = {
    "lm_eval": "lm-eval (for MMLU, MBPP benchmarks)",
    "sentence_transformers": "sentence-transformers (for embedding models)",
    "openai": "openai (for GPT-as-Judge evaluation)",
}
for pkg, desc in optional.items():
    try:
        __import__(pkg)
        print(f"✅ {desc}")
    except ImportError:
        print(f"⚠️ {desc} — not installed (optional)")
'''))

# ============================================================
# STEP 2: DATA EXPLORATION
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""---
## 📊 Step 2: Data Exploration & Validation

Load your CSV, inspect its structure, and validate it for fine-tuning quality.
"""))

cells.append(nbf.v4.new_code_cell('''# Load and preview the dataset
import pandas as pd
from pathlib import Path

data_path = Path(DATA_CSV)

if data_path.exists():
    df = pd.read_csv(data_path)
    print(f"📁 File: {DATA_CSV}")
    print(f"📊 Rows: {len(df):,}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"\\n--- First 3 rows ---")
    df.head(3)
else:
    print(f"❌ File not found: {DATA_CSV}")
    print("   Please update DATA_CSV in the configuration cell above.")
    print("   Supported formats: .csv")
    df = None
'''))

cells.append(nbf.v4.new_code_cell('''# Detailed column analysis
if df is not None:
    # Auto-detect columns
    COLUMN_ALIASES = {
        "instruction": ["instruction", "user", "human", "question", "prompt", "input", "query", "ask"],
        "response": ["response", "assistant", "answer", "output", "reply", "completion", "target", "solution"],
        "system": ["system", "context", "background", "preprompt", "system_prompt"],
        "thinking": ["thinking", "reasoning", "explanation", "chain", "chain_of_thought", "rationale", "thought", "cot"],
        "category": ["category", "domain", "type", "topic", "label", "class"],
    }
    
    detected = {}
    columns_lower = {col.lower().strip(): col for col in df.columns}
    
    for role, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in columns_lower:
                detected[role] = columns_lower[alias]
                break
    
    # Apply user overrides
    if COLUMN_INSTRUCTION: detected["instruction"] = COLUMN_INSTRUCTION
    if COLUMN_RESPONSE: detected["response"] = COLUMN_RESPONSE
    if COLUMN_SYSTEM: detected["system"] = COLUMN_SYSTEM
    if COLUMN_THINKING: detected["thinking"] = COLUMN_THINKING
    
    print("🔍 Auto-detected column mapping:")
    for role, col in detected.items():
        print(f"   {role}: {col}")
    
    required = ["instruction", "response"]
    for req in required:
        if req not in detected:
            print(f"\\n❌ Missing required column: {req}")
            print(f"   Please set COLUMN_{req.upper()} in the configuration cell.")
    
    has_thinking = "thinking" in detected
    if not has_thinking:
        print(f"\\n💡 No thinking/reasoning column detected.")
        print(f"   The pipeline will add CoT scaffolding automatically.")
        if ADD_THINKING_TAGS:
            print(f"   Thinking tags (<think/>...</think/>) will be added to responses.")
else:
    detected = {}
    print("⚠️ No data loaded. Please load your CSV first.")
'''))

cells.append(nbf.v4.new_code_cell('''# Data quality report
if df is not None and detected:
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    
    for role, col in detected.items():
        col_data = df[col]
        
        # Null values
        null_count = col_data.isna().sum()
        # Empty strings
        empty_count = (col_data.astype(str).str.strip() == "").sum()
        # Length stats
        lengths = col_data.dropna().astype(str).str.len()
        
        print(f"\\n📌 {role} -> {col}")
        print(f"   Null values: {null_count} ({null_count/len(df)*100:.1f}%)")
        print(f"   Empty strings: {empty_count}")
        print(f"   Length: min={lengths.min()}, max={lengths.max():,}, mean={lengths.mean():.0f}, median={lengths.median():.0f}")
        
        # Show a sample
        sample_idx = lengths.idxmax()  # Show longest example
        sample = str(col_data.loc[sample_idx])[:300]
        print(f"   Sample (longest): {sample}...")
    
    # Duplicate check
    if "instruction" in detected:
        dup_count = df[detected["instruction"]].duplicated().sum()
        print(f"\\n🔁 Duplicate instructions: {dup_count} ({dup_count/len(df)*100:.1f}%)")
    
    # Dataset size assessment
    n = len(df)
    print(f"\\n📏 Dataset size assessment: {n:,} samples")
    if n < 100:
        print("   ❌ Very small — need 500+ for LoRA fine-tuning")
    elif n < 500:
        print("   ⚠️ Small — recommended 1000+ for best results")
    elif n < 2000:
        print("   ✅ Adequate for LoRA fine-tuning")
    else:
        print("   ✅ Good size for LoRA fine-tuning")
    
    # Code/finance content detection
    code_keywords = ["def ", "class ", "import ", "function", "return ", "```"]
    fin_keywords = ["stock", "portfolio", "risk", "trading", "financial", "revenue", "bank"]
    
    text_col = detected.get("instruction", detected.get("response", ""))
    if text_col:
        code_pct = sum(df[text_col].astype(str).str.contains(kw, regex=False, na=False).sum() for kw in code_keywords) / n * 100
        fin_pct = sum(df[text_col].astype(str).str.lower().str.contains(kw, regex=False, na=False).sum() for kw in fin_keywords) / n * 100
        print(f"\\n💻 Code content: {min(code_pct, 100):.0f}%")
        print(f"🏦 Finance content: {min(fin_pct, 100):.0f}%")
'''))

# ============================================================
# STEP 3: DATASET PREPARATION
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""---
## 🔄 Step 3: Dataset Preparation

Convert your CSV into fine-tuning-ready formats with reasoning augmentation.

The pipeline generates **5 formats** so you can use the same processed data with any model:
- **ChatML** — Qwen, Mistral, DeepSeek models
- **Llama3** — Meta Llama models
- **DeepSeek** — DeepSeek-R1 with `<think/>` tags (best for reasoning!)
- **Alpaca** — Simplest format, universal compatibility
- **ShareGPT** — Conversational format
"""))

cells.append(nbf.v4.new_code_cell('''# Run data preparation
import json
import numpy as np
from pathlib import Path

if df is not None and detected:
    # Clean the data
    initial_len = len(df)
    
    # Remove null/empty instruction or response
    for req_col in ["instruction", "response"]:
        if req_col in detected:
            df = df[df[detected[req_col]].astype(str).str.strip() != ""]
    
    # Remove duplicates
    if "instruction" in detected:
        df = df.drop_duplicates(subset=[detected["instruction"]], keep="first")
    
    # Remove very short responses
    if "response" in detected:
        df = df[df[detected["response"]].astype(str).str.len() >= 10]
    
    df = df.reset_index(drop=True)
    print(f"🧹 Cleaned: {initial_len} -> {len(df)} rows (removed {initial_len - len(df)})")
    
    # Split the data
    np.random.seed(42)
    n = len(df)
    indices = np.random.permutation(n)
    
    train_end = int(n * TRAIN_SPLIT)
    val_end = int(n * (TRAIN_SPLIT + (1 - TRAIN_SPLIT) / 2))
    
    train_df = df.iloc[indices[:train_end]].reset_index(drop=True)
    val_df = df.iloc[indices[train_end:val_end]].reset_index(drop=True)
    test_df = df.iloc[indices[val_end:]].reset_index(drop=True)
    
    print(f"\\n📂 Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Preview a sample
    sample_idx = 0
    inst_col = detected.get("instruction", "")
    resp_col = detected.get("response", "")
    think_col = detected.get("thinking", "")
    
    print(f"\\n--- Sample Training Example ---")
    print(f"Instruction: {str(train_df[inst_col].iloc[sample_idx])[:200]}")
    print(f"Response: {str(train_df[resp_col].iloc[sample_idx])[:200]}")
    if think_col:
        print(f"Thinking: {str(train_df[think_col].iloc[sample_idx])[:200]}")
else:
    print("⚠️ No data loaded. Please complete Step 2 first.")
'''))

cells.append(nbf.v4.new_code_cell('''# Convert to ChatML format (used by Qwen, Mistral, DeepSeek)
def row_to_chatml(row_dict, add_thinking=True, add_cot=True):
    """Convert a row to ChatML format with reasoning augmentation."""
    messages = []
    
    # System message
    system_msg = row_dict.get("system", "")
    if not system_msg:
        system_msg = "You are an expert assistant specializing in coding and fintech. You provide accurate, well-reasoned answers with step-by-step thinking when appropriate."
    
    if add_cot and not system_msg.endswith((".", "!", "?")):
        system_msg += (
            "\\n\\nWhen solving problems, always:\\n"
            "1. Break the problem into steps\\n"
            "2. Show your reasoning process\\n"
            "3. Verify your answer\\n"
            "4. Consider edge cases"
        )
    
    messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": row_dict["instruction"]})
    
    # Assistant response with thinking
    response = row_dict["response"]
    thinking = row_dict.get("thinking", "")
    
    if add_thinking and thinking:
        response = f"<think/>\\n{thinking}\\n</think/>\\n\\n{response}"
    elif add_thinking and not thinking:
        response = f"<think/>\\nLet me think about this step by step.\\n</think/>\\n\\n{response}"
    
    messages.append({"role": "assistant", "content": response})
    
    # Format as ChatML text
    text = "\\n".join(f"<|im_start|>{m[\"role\"]}\\n{m[\"content\"]}<|im_end|>" for m in messages)
    
    return {"messages": messages, "text": text}


# Process all splits
if df is not None and detected:
    processed = {}
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        records = []
        for _, row in split_df.iterrows():
            row_dict = {}
            for role, col in detected.items():
                row_dict[role] = str(row[col]) if pd.notna(row[col]) else ""
            
            record = row_to_chatml(
                row_dict,
                add_thinking=ADD_THINKING_TAGS,
                add_cot=ADD_CHAIN_OF_THOUGHT,
            )
            records.append(record)
        processed[split_name] = records
    
    # Save to disk
    output_dir = Path(DATA_DIR) / "chatml"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, records in processed.items():
        filepath = output_dir / f"{split_name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\\n")
        print(f"💾 Saved {len(records)} records to {filepath}")
    
    # Preview formatted sample
    print(f"\\n--- ChatML Formatted Sample ---")
    print(processed["train"][0]["text"][:800])
    
    print(f"\\n✅ Dataset preparation complete!")
    print(f"   Format: ChatML (with thinking tags: {ADD_THINKING_TAGS})")
    print(f"   Location: {output_dir}")
'''))

# ============================================================
# STEP 4: MODEL SELECTION
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""---
## 🤖 Step 4: Model Selection

Browse all 8 candidate models and choose the one that fits your needs.

**Key considerations:**
- **VRAM budget**: Your GPU has 30GB — all 7B models fit comfortably with QLoRA 4-bit
- **Reasoning priority**: DeepSeek-R1-Distill-Qwen-7B has built-in CoT reasoning
- **Coding priority**: Qwen2.5-Coder models are purpose-built for code
- **14B models**: Fit but tight — require batch_size=1 and reduced LoRA rank
"""))

cells.append(nbf.v4.new_code_cell('''# Model catalog with VRAM analysis
import yaml

MODELS_DIR = Path("./configs/models")

models_info = []
if MODELS_DIR.exists():
    for f in sorted(MODELS_DIR.glob("*.yaml")):
        with open(f) as fh:
            config = yaml.safe_load(fh)
        m = config["model"]
        v = config["vram"]
        l = config["lora"]
        t = config["training"]
        r = config["recommended"]
        models_info.append({
            "short_name": m["short_name"],
            "hf_name": m["name"],
            "size": m["size_billions"],
            "family": m["family"],
            "context": m["context_length"],
            "type": m["type"],
            "train_vram_qlora": v["qlora_4bit_training"],
            "lora_rank": l["rank"],
            "max_seq": t["max_seq_length"],
            "batch_size": t["per_device_train_batch_size"],
            "notes": r["notes"],
        })

# Display as DataFrame
if models_info:
    models_df = pd.DataFrame(models_info)
    print("📋 Available Models:")
    print("=" * 100)
    for _, m in models_df.iterrows():
        vram = m["train_vram_qlora"]
        fit_status = "✅ Good" if "16-18" in str(vram) or "6-8" in str(vram) or "8-10" in str(vram) else ("⚠️ Tight" if "24-28" in str(vram) else "✅ OK")
        print(f"\\n🔑 {m['short_name']}")
        print(f"   HuggingFace: {m['hf_name']}")
        print(f"   Size: {m['size']}B | Context: {m['context']:,} | Type: {m['type']}")
        print(f"   VRAM (QLoRA 4-bit): {vram} | Fit 30GB: {fit_status}")
        print(f"   LoRA Rank: {m['lora_rank']} | Batch Size: {m['batch_size']} | Max Seq: {m['max_seq']:,}")
        print(f"   Notes: {m['notes']}")
else:
    # Fallback if configs not available
    print("📋 Model Catalog (from pipeline configs):")
    catalog = [
        ("qwen2.5-coder-1.5b", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "1.5B", "6-8 GB", "Basic"),
        ("phi-2", "microsoft/phi-2", "2.7B", "8-10 GB", "Moderate"),
        ("qwen2.5-coder-7b", "Qwen/Qwen2.5-Coder-7B-Instruct", "7B", "16-18 GB", "Strong"),
        ("llama-3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "8B", "18-20 GB", "Strong"),
        ("deepseek-coder-7b", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "7B", "16-18 GB", "Strong"),
        ("deepseek-r1-qwen-7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "7B", "16-20 GB", "Exceptional"),
        ("mistral-7b", "mistralai/Mistral-7B-Instruct-v0.3", "7B", "16-18 GB", "Strong"),
        ("qwen2.5-coder-14b", "Qwen/Qwen2.5-Coder-14B-Instruct", "14B", "24-28 GB", "Very Strong"),
    ]
    for short, hf, size, vram, reasoning in catalog:
        print(f"   {short:<30} {size:>5}  VRAM: {vram:<12}  Reasoning: {reasoning}")
'''))

cells.append(nbf.v4.new_code_cell('''# VRAM Budget Check for Selected Model
import torch

if torch.cuda.is_available():
    total_vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    
    # Model-specific VRAM estimates
    vram_estimates = {
        "qwen2.5-coder-1.5b": {"base_4bit": 1.2, "overhead": 5},
        "phi-2": {"base_4bit": 2.0, "overhead": 6},
        "qwen2.5-coder-7b": {"base_4bit": 4.5, "overhead": 12},
        "llama-3.1-8b": {"base_4bit": 5.0, "overhead": 14},
        "deepseek-coder-7b": {"base_4bit": 4.5, "overhead": 12},
        "deepseek-r1-qwen-7b": {"base_4bit": 4.5, "overhead": 14},
        "mistral-7b": {"base_4bit": 4.5, "overhead": 12},
        "qwen2.5-coder-14b": {"base_4bit": 8.5, "overhead": 18},
    }
    
    if MODEL_NAME in vram_estimates:
        est = vram_estimates[MODEL_NAME]
        total_est = est["base_4bit"] + est["overhead"]
        headroom = total_vram - total_est
        
        print(f"🎮 GPU VRAM: {total_vram:.1f} GB")
        print(f"🤖 Model: {MODEL_NAME}")
        print(f"📦 Base model (4-bit): ~{est[\"base_4bit\"]:.1f} GB")
        print(f"⚙️ Training overhead: ~{est[\"overhead\"]} GB")
        print(f"📊 Total estimated: ~{total_est:.1f} GB")
        print(f"📏 Headroom: ~{headroom:.1f} GB")
        
        if headroom >= 10:
            print("\\n✅ COMFORTABLE — Good margin for training")
            print("   You can use larger batch sizes, longer sequences, or LoRA FP16")
        elif headroom >= 4:
            print("\\n✅ FEASIBLE — Sufficient for QLoRA 4-bit training")
            print("   Use recommended batch sizes from the model config")
        elif headroom >= 1:
            print("\\n⚠️ TIGHT — Reduce batch_size to 1, max_seq_length to 2048, LoRA rank to 32")
            print("   Monitor VRAM carefully during training!")
        else:
            print("\\n❌ LIKELY OOM — Model too large for this GPU")
            print("   Consider a smaller model or cloud GPU with more VRAM")
    else:
        print(f"⚠️ No VRAM estimate for {MODEL_NAME}")
else:
    print("⚠️ No GPU detected — cannot check VRAM")
'''))

# ============================================================
# STEP 5: FINE-TUNING
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""---
## 🔥 Step 5: Fine-Tuning with Unsloth + QLoRA

This is the core training step. The cell below will:
1. Load the model in 4-bit quantization
2. Apply LoRA adapter
3. Load and format the dataset
4. Run the training loop
5. Save the fine-tuned model

**⏱️ Expected training times (approximate):**
- 1.5B model: 30-60 min (3 epochs, 1K samples)
- 7B model: 2-4 hours (3 epochs, 1K samples)
- 14B model: 4-8 hours (3 epochs, 1K samples)
"""))

cells.append(nbf.v4.new_code_cell('''# ============================================================
# 🔥 FINE-TUNING — Main Training Cell
# ============================================================

# Map short names to HuggingFace model IDs
MODEL_MAP = {
    "qwen2.5-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "phi-2": "microsoft/phi-2",
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-coder-7b": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-r1-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5-coder-14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
}

# Model-specific defaults
MODEL_DEFAULTS = {
    "qwen2.5-coder-1.5b": {"rank": 64, "alpha": 128, "batch_size": 4, "grad_accum": 4, "seq_len": 4096, "lr": 2e-4},
    "phi-2": {"rank": 64, "alpha": 128, "batch_size": 4, "grad_accum": 4, "seq_len": 2048, "lr": 2e-4},
    "qwen2.5-coder-7b": {"rank": 64, "alpha": 128, "batch_size": 2, "grad_accum": 8, "seq_len": 4096, "lr": 2e-4},
    "llama-3.1-8b": {"rank": 64, "alpha": 128, "batch_size": 2, "grad_accum": 8, "seq_len": 4096, "lr": 1.5e-4},
    "deepseek-coder-7b": {"rank": 64, "alpha": 128, "batch_size": 2, "grad_accum": 8, "seq_len": 4096, "lr": 2e-4},
    "deepseek-r1-qwen-7b": {"rank": 64, "alpha": 128, "batch_size": 2, "grad_accum": 8, "seq_len": 4096, "lr": 1.5e-4},
    "mistral-7b": {"rank": 64, "alpha": 128, "batch_size": 2, "grad_accum": 8, "seq_len": 4096, "lr": 2e-4},
    "qwen2.5-coder-14b": {"rank": 32, "alpha": 64, "batch_size": 1, "grad_accum": 16, "seq_len": 2048, "lr": 1e-4},
}

# Get model ID and defaults
hf_model_id = MODEL_MAP.get(MODEL_NAME, MODEL_NAME)
defaults = MODEL_DEFAULTS.get(MODEL_NAME, {"rank": 64, "alpha": 128, "batch_size": 2, "grad_accum": 8, "seq_len": 4096, "lr": 2e-4})

# Apply user overrides
rank = LORA_RANK or defaults["rank"]
alpha = rank * 2  # Standard: alpha = 2 * rank
batch_size = BATCH_SIZE or defaults["batch_size"]
grad_accum = defaults["grad_accum"]
seq_len = MAX_SEQ_LENGTH or defaults["seq_len"]
lr = LEARNING_RATE or defaults["lr"]

print(f"🤖 Model: {hf_model_id}")
print(f"⚙️ Config: rank={rank}, alpha={alpha}, batch={batch_size}, seq_len={seq_len}, lr={lr}")
print(f"📁 Data: {DATA_DIR}/chatml/train.jsonl")
'''))

cells.append(nbf.v4.new_code_cell('''# ============================================================
# Load Model with Unsloth
# ============================================================

# Try Unsloth first, fall back to standard loading
use_unsloth = False
try:
    from unsloth import FastLanguageModel
    use_unsloth = True
    print("✅ Using Unsloth for 2x faster training")
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print("⚠️ Unsloth not available, using standard transformers")

if use_unsloth:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_model_id,
        max_seq_length=seq_len,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
        trust_remote_code=MODEL_NAME in ["phi-2", "deepseek-coder-7b", "deepseek-r1-qwen-7b"],
    )
else:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

total_params = sum(p.numel() for p in model.parameters())
print(f"\\n✅ Model loaded!")
print(f"   Parameters: {total_params/1e9:.2f}B")
print(f"   Vocab size: {tokenizer.vocab_size:,}")

# Check current VRAM usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"   VRAM allocated: {allocated:.1f} GB")
    print(f"   VRAM reserved: {reserved:.1f} GB")
'''))

cells.append(nbf.v4.new_code_cell('''# ============================================================
# Apply LoRA Adapter
# ============================================================

if use_unsloth:
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
else:
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA applied!")
print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
print(f"   Rank: {rank}, Alpha: {alpha}")
'''))

cells.append(nbf.v4.new_code_cell('''# ============================================================
# Load Dataset
# ============================================================

from datasets import load_from_disk, Dataset

data_path = Path(DATA_DIR) / "chatml"

# Load JSONL files
train_file = data_path / "train.jsonl"
val_file = data_path / "val.jsonl"

if train_file.exists():
    train_dataset = Dataset.from_list(
        [json.loads(line) for line in open(train_file)]
    )
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    
    if val_file.exists():
        eval_dataset = Dataset.from_list(
            [json.loads(line) for line in open(val_file)]
        )
        print(f"✅ Val dataset: {len(eval_dataset)} samples")
    else:
        eval_dataset = None
        print("⚠️ No validation dataset found")
else:
    print(f"❌ Training data not found at {train_file}")
    print("   Please run Step 3 (Dataset Preparation) first.")
    train_dataset = None
    eval_dataset = None
'''))

cells.append(nbf.v4.new_code_cell('''# ============================================================
# 🚀 START TRAINING
# ============================================================
# ⏱️ This will take a while — grab a coffee!
# Monitor VRAM with: nvidia-smi -l 5

from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import time

if train_dataset is not None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_output_dir = f"{OUTPUT_DIR}/{MODEL_NAME}_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.1,
        num_train_epochs=EPOCHS,
        learning_rate=lr,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="tensorboard",
        seed=42,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=seq_len,
        packing=True,
        args=training_args,
    )
    
    # Show GPU stats before training
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    print(f"\\n🚀 Starting training...")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Samples: {len(train_dataset)}")
    print(f"   Batch size: {batch_size} x {grad_accum} accum = {batch_size * grad_accum} effective")
    print(f"   Learning rate: {lr}")
    print(f"   Output: {run_output_dir}")
    print(f"\\n⏱️ Estimated time: {(len(train_dataset) * EPOCHS) / (batch_size * grad_accum) * 2 / 3600:.1f} - {(len(train_dataset) * EPOCHS) / (batch_size * grad_accum) * 5 / 3600:.1f} hours")
    
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    print(f"\\n✅ Training complete!")
    print(f"   Time: {training_time/3600:.2f} hours")
    print(f"   Loss: {train_result.training_loss:.4f}")
    
    # Save model
    trainer.save_model(run_output_dir)
    print(f"   Saved to: {run_output_dir}")
else:
    print("❌ No training data available. Run Step 3 first.")
'''))

# ============================================================
# STEP 6: EVALUATION
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""---
## 📏 Step 6: Evaluation

Evaluate your fine-tuned model on multiple benchmarks. Each benchmark tests a different capability:

| Benchmark | What it Tests |
|---|---|
| **MMLU** | General knowledge (57 subjects, includes finance) |
| **GSM8K** | Mathematical reasoning |
| **HumanEval** | Code generation (Python) |
| **MT-Bench** | Multi-turn conversation (GPT-judged) |
| **FinQA** | Financial question answering |
| **Perplexity** | Language model quality (lower = better) |
| **Fintech Custom** | Domain-specific fintech tasks |
"""))

cells.append(nbf.v4.new_code_cell('''# ============================================================
# Quick Inference Test — Before running full benchmarks
# ============================================================

# Test the model with a sample prompt
test_prompts = [
    "Write a Python function to calculate the Sharpe ratio of a portfolio.",
    "Explain the key requirements of PSD2 for payment service providers in the EU.",
    "Design a fraud detection system for credit card transactions using isolation forests.",
    "Implement a simple moving average crossover trading strategy in Python.",
]

# Enable inference mode
if use_unsloth:
    FastLanguageModel.for_inference(model)

for prompt in test_prompts[:2]:  # Test first 2 prompts
    print(f"\\n{'='*60}")
    print(f"📝 Prompt: {prompt}")
    print(f"{'='*60}")
    
    # Format prompt with chat template
    messages = [{"role": "user", "content": prompt}]
    try:
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    except Exception:
        inputs = tokenizer(f"<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n", return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs if inputs.dim() == 2 else inputs.unsqueeze(0) if inputs.dim() == 1 else inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.shape[-1]:] if inputs.dim() == 2 else outputs[0], skip_special_tokens=True)
    print(f"🤖 Response: {response[:500]}...")
'''))

cells.append(nbf.v4.new_code_cell('''# ============================================================
# Full Benchmark Evaluation
# ============================================================
# Note: This runs all benchmarks which can take hours.
# For quick evaluation, set BENCHMARKS = ["gsm8k", "fintech"]

# Run evaluation using the pipeline script
# Uncomment the benchmark you want to run:

# --- Option 1: Quick eval (2-3 benchmarks, ~30 min) ---
# BENCHMARKS = ["gsm8k", "fintech"]

# --- Option 2: Full eval (all benchmarks, 4-8 hours) ---
# BENCHMARKS = ["perplexity", "mmlu", "gsm8k", "humaneval", "mt_bench", "fintech"]

# --- Option 3: Run via pipeline script ---
# !python evaluation/evaluate.py \\
#     --model {run_output_dir if 'run_output_dir' in dir() else OUTPUT_DIR} \\
#     --benchmarks gsm8k fintech \\
#     --output ./eval_results

print("📏 To run full evaluation, execute:")
print(f"   python evaluation/evaluate.py --model {OUTPUT_DIR}/{MODEL_NAME} --benchmarks all --output ./eval_results")
print()
print("🔬 For quick evaluation (recommended first):")
print(f"   python evaluation/evaluate.py --model {OUTPUT_DIR}/{MODEL_NAME} --benchmarks gsm8k fintech --output ./eval_results")
'''))

# ============================================================
# STEP 7: MODEL COMPARISON
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""---
## 📊 Step 7: Model Comparison & Recommendations

After evaluating multiple models, compare their performance to find the best fit.

Run this after you've fine-tuned and evaluated at least 2 models.
"""))

cells.append(nbf.v4.new_code_cell('''# ============================================================
# Compare Models
# ============================================================

# Run comparison using the pipeline script
# !python compare_models.py --results-dir ./eval_results --priority reasoning

# Or view the built-in model profiles and recommendations
print("🏆 MODEL RECOMMENDATIONS BY PRIORITY")
print("=" * 70)

recommendations = {
    "🧠 Reasoning & Thinking": {
        "top": "deepseek-r1-qwen-7b",
        "reason": "Built-in chain-of-thought, self-reflection, <think/> blocks. Distilled from R1 reasoning model.",
        "alt": "qwen2.5-coder-7b (strong reasoning + best coding)",
    },
    "💻 Coding": {
        "top": "qwen2.5-coder-7b",
        "reason": "Best-in-class 7B coding model. Beats many larger models on code benchmarks. 128K context.",
        "alt": "deepseek-coder-7b (MoE efficiency)",
    },
    "🏦 Fintech Analysis": {
        "top": "deepseek-r1-qwen-7b",
        "reason": "Exceptional reasoning for financial analysis. Multi-step problem solving. Self-verification.",
        "alt": "llama-3.1-8b (strong general fintech)",
    },
    "⚖️ Balanced": {
        "top": "qwen2.5-coder-7b",
        "reason": "Best all-around 7B model. Strong coding + good reasoning + comfortable VRAM margin.",
        "alt": "mistral-7b (battle-tested, efficient)",
    },
    "🚀 Maximum Quality": {
        "top": "qwen2.5-coder-14b",
        "reason": "Highest benchmark scores. TIGHT on 30GB VRAM — QLoRA 4-bit only, batch=1.",
        "alt": "qwen2.5-coder-7b (comfortable VRAM)",
    },
    "⚡ Fast Prototyping": {
        "top": "qwen2.5-coder-1.5b",
        "reason": "Tiny, fast, good coding. Ideal for rapid iteration and experiments.",
        "alt": "phi-2 (good reasoning for size)",
    },
}

for priority, info in recommendations.items():
    print(f"\\n{priority}")
    print(f"   🥇 {info[\"top\"]}")
    print(f"      Why: {info[\"reason\"]}")
    print(f"   🥈 {info[\"alt\"]}")
'''))

# ============================================================
# STEP 8: EXPORT & DEPLOY
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""---
## 📤 Step 8: Export & Deploy

Export your fine-tuned model in different formats for various deployment scenarios:

| Format | Use Case | Size |
|---|---|---|
| **LoRA Adapter** | Deploy with base model (smallest) | ~50-200 MB |
| **Merged 16-bit** | vLLM, TGI, HuggingFace serving | ~14-28 GB |
| **Merged 4-bit** | Smaller deployment, still good quality | ~4-9 GB |
| **GGUF** | llama.cpp, Ollama, local inference | ~3-8 GB |
"""))

cells.append(nbf.v4.new_code_cell('''# ============================================================
# Export Model
# ============================================================

# Choose export format based on your deployment target:

export_dir = f"{OUTPUT_DIR}/{MODEL_NAME}"

# --- Option 1: LoRA adapter only (smallest, requires base model) ---
if use_unsloth:
    # model.save_pretrained(f"{export_dir}/lora_adapter")
    # tokenizer.save_pretrained(f"{export_dir}/lora_adapter")
    print("💾 LoRA adapter export: model.save_pretrained(path)")
    print("   Small size (~50-200MB), requires base model at inference time")

# --- Option 2: Merged 16-bit (for vLLM, TGI) ---
if use_unsloth:
    # model.save_pretrained_merged(f"{export_dir}/merged_16bit", tokenizer, save_method="merged_16bit")
    print("💾 Merged 16-bit export: model.save_pretrained_merged(path, tokenizer, save_method='merged_16bit')")
    print("   Full model, best quality, for server deployment (vLLM, TGI)")

# --- Option 3: GGUF (for llama.cpp, Ollama) ---
if use_unsloth:
    # model.save_pretrained_gguf(f"{export_dir}/gguf", tokenizer, quantization_method="q4_k_m")
    print("💾 GGUF export: model.save_pretrained_gguf(path, tokenizer, quantization_method='q4_k_m')")
    print("   For local inference with llama.cpp or Ollama")

# --- Option 4: Merged 4-bit (smaller deployment) ---
if use_unsloth:
    # model.save_pretrained_merged(f"{export_dir}/merged_4bit", tokenizer, save_method="merged_4bit_forced")
    print("💾 Merged 4-bit export: model.save_pretrained_merged(path, tokenizer, save_method='merged_4bit_forced')")
    print("   4-bit quantized merged model, good balance of size and quality")

print("\\n⚠️ Uncomment the export line you need above to actually export the model!")
print("\\n🚀 Deployment examples:")
print("   Ollama:  ollama create mymodel -f Modelfile")
print("   vLLM:    python -m vllm.entrypoints.openai.api_server --model ./merged_16bit")
print("   HF Hub:  huggingface-cli upload your-repo ./merged_16bit")
'''))

# ============================================================
# STEP 9: EMBEDDING MODELS
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""---
## 🔍 Step 9: Embedding Models (NLP)

Fine-tune and evaluate embedding models for RAG and semantic search in the fintech domain.

**Recommended:** BAAI/bge-large-en-v1.5 (1024-dim, best for fintech semantic search)
"""))

cells.append(nbf.v4.new_code_cell('''# ============================================================
# Embedding Model Pipeline
# ============================================================

# List available embedding models
embedding_models = {
    "bge-large": {"name": "BAAI/bge-large-en-v1.5", "dim": 1024, "max_len": 512, "vram": "~2 GB", "rec": "Top Pick"},
    "e5-large": {"name": "intfloat/e5-large-v2", "dim": 1024, "max_len": 512, "vram": "~2 GB", "rec": "Alternative"},
    "bge-m3": {"name": "BAAI/bge-m3", "dim": 1024, "max_len": 8192, "vram": "~3 GB", "rec": "Long Docs"},
    "minilm": {"name": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384, "max_len": 256, "vram": "~0.5 GB", "rec": "Baseline"},
}

print("🔍 Available Embedding Models:")
print("-" * 80)
for key, info in embedding_models.items():
    print(f"  {key:<12} {info['name']:<40} dim={info['dim']}  VRAM={info['vram']}  [{info['rec']}]")

print("\\n💡 To fine-tune an embedding model:")
print("   python evaluation/embedding_pipeline.py --model bge-large --data ./data/processed/chatml/train.jsonl --train")
print("\\n💡 To compare embedding models:")
print("   python evaluation/embedding_pipeline.py --compare --models bge-large e5-large minilm")
'''))

cells.append(nbf.v4.new_code_cell('''# Quick embedding evaluation
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Load model
    emb_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    
    # Test with fintech queries
    queries = [
        "How do I assess credit risk for P2P lending?",
        "Implement a moving average crossover strategy in Python",
        "What are SOX compliance requirements for fintech?",
    ]
    
    docs = [
        "Credit risk assessment involves evaluating borrower creditworthiness through scoring models and debt-to-income analysis.",
        "Moving average crossover generates buy/sell signals when short MA crosses above/below long MA.",
        "SOX requires internal controls, audit committees, and CEO/CFO certification of financial reports.",
    ]
    
    q_embeddings = emb_model.encode(queries)
    d_embeddings = emb_model.encode(docs)
    
    sims = cosine_similarity(q_embeddings, d_embeddings)
    
    print("📊 Embedding Retrieval Test:")
    for i, query in enumerate(queries):
        top_idx = sims[i].argmax()
        top_sim = sims[i].max()
        correct = "✅" if top_idx == i else "❌"
        print(f"  {correct} Query: {query[:50]}...")
        print(f"     Top doc: [{top_idx}] (sim={top_sim:.3f})")
    
    accuracy = sum(1 for i in range(len(queries)) if sims[i].argmax() == i) / len(queries)
    print(f"\\n🎯 Retrieval Accuracy: {accuracy:.0%}")
    
except ImportError:
    print("⚠️ sentence-transformers not installed. Install with:")
    print("   pip install sentence-transformers scikit-learn")
'''))

# ============================================================
# STEP 10: SUMMARY
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""---
## 📋 Summary & Next Steps

### What You've Built
✅ A complete fine-tuning pipeline for open-source LLMs in the coding + fintech domain
✅ Support for reasoning fine-tuning (CoT, self-reflection, thinking tags)
✅ 8 candidate models analyzed for your 30GB VRAM constraint
✅ Multi-benchmark evaluation framework
✅ Embedding model pipeline for RAG applications
✅ Export to multiple deployment formats

### Recommended Workflow
1. **Start small**: Fine-tune `qwen2.5-coder-1.5b` first to validate your data and pipeline
2. **Scale up**: Move to `deepseek-r1-qwen-7b` for reasoning or `qwen2.5-coder-7b` for coding
3. **Evaluate**: Run benchmarks after each fine-tuning run
4. **Compare**: Use the comparison tool to find the best model for your use case
5. **Deploy**: Export as GGUF for local use or merged 16-bit for server deployment

### Quick Commands Reference
```bash
# Validate data
python data/data_validator.py --input your_data.csv

# Prepare dataset
python data/prepare_dataset.py --input your_data.csv --output-dir ./data/processed

# Fine-tune (interactive)
python run_pipeline.py --interactive

# Fine-tune (direct)
python training/finetune.py --model deepseek-r1-qwen-7b --data ./data/processed/chatml

# Evaluate
python evaluation/evaluate.py --model ./outputs/deepseek-r1-qwen-7b --benchmarks all

# Compare
python compare_models.py --results-dir ./eval_results --priority reasoning
```

### Files in This Pipeline
| Directory | Purpose |
|---|---|
| `configs/models/` | YAML configs for 8 models |
| `configs/training/` | QLoRA and LoRA training configs |
| `configs/evaluation/` | Benchmark and fintech eval configs |
| `data/` | Dataset preparation and validation scripts |
| `training/` | Fine-tuning and model merge scripts |
| `evaluation/` | Benchmark evaluation and embedding pipeline |
| `utils/` | GPU monitoring utility |
| `run_pipeline.py` | Main orchestrator (all stages) |
| `compare_models.py` | Model comparison engine |
| `requirements.txt` | Python dependencies |
"""))

# Add all cells to notebook
nb.cells = cells

# Save
output_path = "/home/z/my-project/download/llm-finetune-pipeline/LLM_FineTuning_Pipeline.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook saved to: {output_path}")
