#  LLM Fine-Tuning Pipeline

**Fine-tune and evaluate open-source LLMs for Coding & Fintech — with Reasoning & Thinking support**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-Required-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Unsloth](https://img.shields.io/badge/Unsloth-2x%20Faster-orange.svg)](https://github.com/unslothai/unsloth)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

##  What This Pipeline Does

This is a **complete, end-to-end pipeline** for fine-tuning and evaluating open-source Large Language Models (LLMs) from Hugging Face, specifically designed for:

- **Coding** — code generation, debugging, refactoring, explanation
- **Fintech** — regulatory compliance, risk assessment, algorithmic trading, fraud detection, portfolio optimization
- **Reasoning & Thinking** — chain-of-thought (CoT), self-reflection, tool use, multi-step problem solving

Plus an **embedding model pipeline** for RAG and semantic search in the fintech domain.

### Key Features

-  **8 candidate models** pre-configured (1.5B → 14B) — pick what fits your GPU
-  **QLoRA/LoRA + Unsloth** — 2x faster training, fits 30GB VRAM
-  **Reasoning fine-tuning** — `<think/>` tags, CoT scaffolding, self-reflection prompts
-  **7 benchmarks** — MMLU, GSM8K, HumanEval, MBPP, MT-Bench, FinQA, ConvFinQA
-  **GPT-as-Judge** — automated quality scoring with OpenAI models
-  **Custom fintech eval** — regulatory, risk, trading, fraud, portfolio prompts
-  **Embedding models** — fine-tune BGE/E5 for fintech RAG
-  **Multiple export formats** — LoRA adapter, merged 16-bit, GGUF (for Ollama)
-  **Streamlit Web UI** — interactive dashboard for the entire pipeline
-  **Jupyter notebook** — interactive step-by-step walkthrough
-  **GPU monitoring** — real-time VRAM tracking during training

---

##  Model Recommendations (30GB VRAM)

| Priority |  Top Pick |  Alternative | Why |
|----------|------------|----------------|-----|
| **Reasoning & Thinking** | DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-Coder-7B | Built-in CoT + self-reflection + `<think/>` blocks |
| **Coding** | Qwen2.5-Coder-7B-Instruct | DeepSeek-Coder-V2-Lite | Best 7B coding benchmarks, 128K context |
| **Fintech Analysis** | DeepSeek-R1-Distill-Qwen-7B | Llama-3.1-8B | Best reasoning for complex finance tasks |
| **Balanced** | Qwen2.5-Coder-7B-Instruct | Mistral-7B-v0.3 | Strong coding + solid reasoning + comfortable VRAM |
| **Max Quality** | Qwen2.5-Coder-14B-Instruct | Qwen2.5-Coder-7B | Highest scores (tight VRAM, batch=1) |
| **Fast Prototyping** | Qwen2.5-Coder-1.5B-Instruct | Phi-2 (2.7B) | Quick iteration, low VRAM |
| **Embedding / RAG** | BAAI/bge-large-en-v1.5 | intfloat/e5-large-v2 | Best fintech semantic search |

---

##  Project Structure

```
llm-finetune-pipeline/
├── configs/
│   ├── models/                        # YAML configs for each model
│   │   ├── qwen2.5-coder-1.5b.yaml
│   │   ├── phi-2.yaml
│   │   ├── qwen2.5-coder-7b.yaml
│   │   ├── llama-3.1-8b.yaml
│   │   ├── deepseek-coder-7b.yaml
│   │   ├── deepseek-r1-distill-qwen-7b.yaml
│   │   ├── mistral-7b.yaml
│   │   └── qwen2.5-coder-14b.yaml
│   ├── training/
│   │   ├── qlora_4bit.yaml            # QLoRA 4-bit config (recommended)
│   │   └── lora_fp16.yaml             # LoRA FP16 config (small models only)
│   └── evaluation/
│       ├── benchmarks.yaml            # Benchmark configuration
│       └── fintech_prompts.json       # Custom fintech evaluation prompts
├── data/
│   ├── prepare_dataset.py             # CSV → fine-tuning format converter
│   └── data_validator.py              # CSV quality validation
├── training/
│   ├── finetune.py                    # Unsloth + LoRA/QLoRA training
│   └── merge_adapter.py              # LoRA adapter merger
├── evaluation/
│   ├── evaluate.py                    # Full benchmark evaluation
│   ├── embedding_pipeline.py          # Embedding model pipeline
│   └── benchmarks/                    # Individual benchmark scripts
├── utils/
│   └── gpu_monitor.py                 # VRAM monitoring utility
├── run_pipeline.py                    # Main orchestrator (all stages)
├── compare_models.py                  # Model comparison engine
├── app.py                             # Streamlit web UI
├── LLM_FineTuning_Pipeline.ipynb     # Interactive Jupyter notebook
├── requirements.txt                   # Python dependencies
├── quickstart.sh                      # Quick start script
└── README.md                          # This file
```

---

##  Quick Start

### Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** with 30GB+ VRAM (A100, A6000, RTX 4090, V100-32GB, etc.)
- **CUDA 11.8+** and **cuDNN**

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/llm-finetune-pipeline.git
cd llm-finetune-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install Unsloth (for 2x faster training)
pip install unsloth
```

### Option A: Streamlit Web UI (Easiest!)

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` — you'll get an interactive dashboard with:
-  Data upload and validation
-  Model selection with VRAM check
-  Fine-tuning configuration
-  Benchmark evaluation setup
-  Model comparison and recommendations
-  Export and deployment guides

### Option B: Interactive Notebook

```bash
jupyter lab LLM_FineTuning_Pipeline.ipynb
```

Follow the step-by-step cells — start with Step 0 to configure your setup.

### Option C: Command Line

```bash
# Step 1: Validate your CSV dataset
python data/data_validator.py --input your_data.csv

# Step 2: Prepare dataset (auto-detects columns, converts to 5 formats)
python data/prepare_dataset.py --input your_data.csv --output-dir ./data/processed

# Step 3: Fine-tune a model
python training/finetune.py --model deepseek-r1-qwen-7b --data ./data/processed/chatml

# Step 4: Evaluate
python evaluation/evaluate.py --model ./outputs/deepseek-r1-qwen-7b --benchmarks all

# Step 5: Compare models
python compare_models.py --results-dir ./eval_results --priority reasoning
```

### Option D: Full Pipeline (One Command)

```bash
# Run the entire pipeline end-to-end
python run_pipeline.py --model deepseek-r1-qwen-7b --data your_data.csv

# Quick test run (1 epoch, 100 samples)
python run_pipeline.py --model qwen2.5-coder-1.5b --data your_data.csv --test-run
```

### Option E: Interactive Mode

```bash
python run_pipeline.py --interactive
```

---

##  Dataset Preparation

### CSV Format

Your CSV must have at minimum an **instruction** column and a **response** column. The pipeline auto-detects common column names:

| Role | Recognized Column Names | Required? |
|------|------------------------|-----------|
| Instruction | `instruction`, `user`, `human`, `question`, `prompt`, `input`, `query` | ✅ Yes |
| Response | `response`, `assistant`, `answer`, `output`, `reply`, `completion` | ✅ Yes |
| System | `system`, `context`, `background`, `system_prompt` | Optional |
| Thinking | `thinking`, `reasoning`, `explanation`, `chain_of_thought`, `rationale`, `cot` | Optional* |
| Category | `category`, `domain`, `type`, `topic`, `label` | Optional |

> **\*Thinking column**: If you have reasoning traces, the pipeline wraps them in `<think/>...</think/>` tags for DeepSeek-R1 style reasoning fine-tuning. If not present, CoT scaffolding is added automatically.

### Example CSV

```csv
instruction,response,thinking,category
"Write a Python function to calculate Sharpe ratio","def sharpe_ratio(returns, risk_free_rate=0.0): ...","First, I need to understand the Sharpe ratio formula: (mean_return - risk_free) / std_return. Then implement it step by step...","financial_analysis"
"Explain PSD2 SCA requirements","PSD2 Strong Customer Authentication requires...","Let me break this down: PSD2 is an EU directive, SCA has three elements (something you know, have, are)...","regulatory_compliance"
```

### Custom Column Names

```bash
python data/prepare_dataset.py \
    --input your_data.csv \
    --columns instruction=YourQuestionCol response=YourAnswerCol thinking=YourReasoningCol
```

---

##  Fine-Tuning

### Model Selection

List all available models:

```bash
python training/finetune.py --list-models
```

### Training Commands

```bash
# Fine-tune with defaults (QLoRA 4-bit, Unsloth)
python training/finetune.py --model deepseek-r1-qwen-7b --data ./data/processed/chatml

# Override training parameters
python training/finetune.py --model qwen2.5-coder-7b --data ./data/processed/chatml \
    --epochs 5 --lr 1.5e-4 --rank 32 --max-seq-length 2048

# Quick test run (1 epoch, 100 samples)
python training/finetune.py --model qwen2.5-coder-1.5b --data ./data/processed/chatml --test-run

# Use LoRA FP16 instead of QLoRA (small models only!)
python training/finetune.py --model qwen2.5-coder-1.5b --data ./data/processed/chatml \
    --training-config lora_fp16
```

### Reasoning Fine-Tuning Options

The pipeline supports several reasoning augmentation strategies, configured in the data preparation step:

| Strategy | What It Does | When to Use |
|----------|-------------|-------------|
| **`<think/>` tags** | Wraps reasoning in thinking blocks | You have a thinking/reasoning column |
| **CoT scaffolding** | Adds step-by-step instructions to system prompt | No thinking column available |
| **Self-reflection** | Adds reflection prompts after responses | You want self-correction behavior |
| **All combined** | Thinking tags + CoT + reflection | Maximum reasoning capability |

### Export Formats

```bash
# After training, export in multiple formats
python training/finetune.py --model deepseek-r1-qwen-7b --data ./data/processed/chatml \
    --export lora merged_16bit gguf

# Or merge adapter separately
python training/merge_adapter.py \
    --base deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --adapter ./outputs/deepseek-r1-qwen-7b \
    --output ./merged_model \
    --method merged_16bit
```

| Format | Size | Use Case |
|--------|------|----------|
| `lora` | ~50-200 MB | Smallest, requires base model at inference |
| `merged_16bit` | ~14-28 GB | vLLM, TGI, HuggingFace serving |
| `merged_4bit` | ~4-9 GB | Smaller deployment, good quality |
| `gguf` | ~3-8 GB | llama.cpp, Ollama, local inference |

---

##  Evaluation

### Benchmark Suite

```bash
# Full evaluation (all benchmarks)
python evaluation/evaluate.py --model ./outputs/deepseek-r1-qwen-7b --benchmarks all

# Quick evaluation (selected benchmarks)
python evaluation/evaluate.py --model ./outputs/deepseek-r1-qwen-7b --benchmarks gsm8k fintech

# Compare fine-tuned vs baseline
python evaluation/evaluate.py --model ./outputs/deepseek-r1-qwen-7b \
    --baseline deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --benchmarks all
```

| Benchmark | Category | Key Metric | Relevance |
|-----------|----------|------------|-----------|
| **MMLU** | General Knowledge | acc_norm | 57 subjects incl. finance & CS |
| **GSM8K** | Math Reasoning | acc | Fintech calculations |
| **HumanEval** | Code Generation | pass@1 | Core coding benchmark |
| **MBPP** | Python Coding | pass@1 | Coding breadth |
| **MT-Bench** | Conversation | score (1-10) | GPT-judged quality |
| **FinQA** | Financial QA | acc | Numerical finance reasoning |
| **ConvFinQA** | Conversational Finance | acc | Multi-turn finance |
| **Perplexity** | Language Quality | ppl | Lower = better |
| **Fintech Custom** | Domain-Specific | GPT-judged | 6 categories, 24 prompts |

### GPT-as-Judge

Set your OpenAI API key for automated quality scoring:

```bash
export OPENAI_API_KEY="sk-..."
python evaluation/evaluate.py --model ./outputs/deepseek-r1-qwen-7b --benchmarks mt_bench fintech
```

### Custom Fintech Evaluation

The pipeline includes 24 custom fintech prompts across 6 categories:

-  **Regulatory Compliance** — PSD2, SOX, AML/KYC, Dodd-Frank
-  **Risk Assessment** — credit risk, fraud, market risk, operational risk
-  **Financial Analysis** — Sharpe ratio, DCF, WACC, current ratio
-  **Algorithmic Trading** — MA crossover, Bollinger Bands, VWAP, momentum
-  **Fraud Detection** — feature engineering, isolation forest, graph-based
-  **Portfolio Optimization** — Markowitz, CAPM, risk parity, Black-Litterman

---

##  Embedding Models

Fine-tune and evaluate embedding models for fintech RAG and semantic search:

```bash
# Fine-tune an embedding model
python evaluation/embedding_pipeline.py --model bge-large --data ./data/processed/chatml/train.jsonl --train

# Evaluate embedding quality
python evaluation/embedding_pipeline.py --model bge-large --evaluate

# Compare multiple embedding models
python evaluation/embedding_pipeline.py --compare --models bge-large e5-large minilm
```

| Model | Dimension | Max Length | VRAM | Best For |
|-------|-----------|------------|------|----------|
| **bge-large** | 1024 | 512 | ~2 GB |  Semantic search, RAG |
| e5-large | 1024 | 512 | ~2 GB | Text similarity |
| bge-m3 | 1024 | 8192 | ~3 GB | Long documents |
| minilm | 384 | 256 | ~0.5 GB | Fast prototyping |

---

##  VRAM Reference

| Model | Size | Base (4-bit) | QLoRA Training | Fit 30GB? | Risk |
|-------|------|-------------|----------------|-----------|------|
| qwen2.5-coder-1.5b | 1.5B | ~1.2 GB | 6-8 GB |  Comfortable | None |
| phi-2 | 2.7B | ~2.0 GB | 8-10 GB |  Comfortable | None |
| qwen2.5-coder-7b | 7B | ~4.5 GB | 16-18 GB |  Good margin | Low |
| llama-3.1-8b | 8B | ~5.0 GB | 18-20 GB |  QLoRA only | Low |
| deepseek-coder-7b | 7B | ~4.5 GB | 16-18 GB |  Good margin | Low |
| deepseek-r1-qwen-7b | 7B | ~4.5 GB | 16-20 GB |  Good margin | Low |
| mistral-7b | 7B | ~4.5 GB | 16-18 GB |  Good margin | Low |
| qwen2.5-coder-14b | 14B | ~8.5 GB | 24-28 GB |  Tight | **HIGH** |

> **14B Warning**: Requires `batch_size=1`, `lora_rank=32`, `max_seq_length=2048`. Monitor VRAM carefully!

---

##  Troubleshooting

### Out of Memory (OOM)

```bash
# If OOM during 14B training:
# 1. Reduce sequence length to 1024
# 2. Reduce LoRA rank to 16
# 3. Reduce batch size to 1 (increase gradient_accumulation_steps)

# If OOM during 7B training:
# 1. Reduce batch_size to 1
# 2. Set max_seq_length to 2048
# 3. Ensure gradient_checkpointing is enabled
```

### Slow Training

```bash
# Verify Unsloth is installed
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"

# Check GPU utilization
nvidia-smi -l 5

# Run GPU monitor in background
python utils/gpu_monitor.py --interval 30 --log-file gpu_log.jsonl
```

### Dataset Issues

```bash
# Run the validator first
python data/data_validator.py --input your_data.csv --output validation_report.json

# If columns aren't detected, specify them explicitly
python data/prepare_dataset.py --input your_data.csv \
    --columns instruction=YourCol response=YourCol
```

---

##  LoRA Hyperparameter Reference

| Parameter | Default (7B) | Default (14B) | Range | Description |
|-----------|-------------|---------------|-------|-------------|
| LoRA Rank | 64 | 32 | 8-128 | Higher = more capacity, more VRAM |
| LoRA Alpha | 128 | 64 | r to 2r | Standard: alpha = 2 × rank |
| LoRA Dropout | 0.05 | 0.05 | 0-0.1 | Increase to 0.1 if overfitting |
| Learning Rate | 2e-4 | 1e-4 | 1e-5 to 5e-4 | Higher for smaller models |
| Batch Size | 2 | 1 | 1-8 | Use grad_accum to compensate |
| Grad Accumulation | 8 | 16 | 4-16 | Effective batch = batch × accum |
| Max Seq Length | 4096 | 2048 | 512-8192 | Longer = more VRAM |
| Epochs | 3 | 3 | 1-5 | Monitor eval loss for early stopping |

---

##  Deployment Examples

### Ollama (Local Inference)

```bash
# Export as GGUF
python training/finetune.py --model deepseek-r1-qwen-7b --data ./data/processed/chatml --export gguf

# Create Ollama model
# Create a Modelfile:
# FROM ./outputs/deepseek-r1-qwen-7b/gguf/model-q4_k_m.gguf
ollama create fintech-coder -f Modelfile
ollama run fintech-coder
```

### vLLM (Server Deployment)

```bash
# Export as merged 16-bit
python training/finetune.py --model deepseek-r1-qwen-7b --data ./data/processed/chatml --export merged_16bit

# Serve with vLLM
python -m vllm.entrypoints.openai.api_server \
    --model ./outputs/deepseek-r1-qwen-7b/merged_16bit \
    --host 0.0.0.0 --port 8000
```

### Hugging Face Hub

```bash
huggingface-cli upload your-username/fintech-coder ./outputs/deepseek-r1-qwen-7b/merged_16bit
```

---

##  License

This pipeline is released under the MIT License. Individual models have their own licenses — please check before deployment:

| Model | License |
|-------|---------|
| Qwen2.5-Coder | Apache 2.0 |
| Phi-2 | MIT |
| Llama-3.1 | Llama 3.1 Community License |
| DeepSeek-Coder | DeepSeek License |
| DeepSeek-R1 | MIT |
| Mistral | Apache 2.0 |

---

##  Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) — 2x faster LLM fine-tuning
- [Hugging Face](https://huggingface.co/) — Model hub and transformers library
- [TRL](https://github.com/huggingface/trl) — Transformer Reinforcement Learning
- [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) — Evaluation harness
- [sentence-transformers](https://www.sbert.net/) — Embedding models

---

**Built for the coding + fintech domain. Fine-tune reasoning, not just answers.** 🧠
