"""
LLM Fine-Tuning Pipeline - Streamlit Web Interface
===================================================
A complete interactive UI for fine-tuning and evaluating LLMs.
Run with: streamlit run app.py
"""

import streamlit as st
import yaml
import json
import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="LLM Fine-Tuning Pipeline",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="expanded",
)

PIPELINE_DIR = Path(__file__).parent
MODELS_DIR = PIPELINE_DIR / "configs" / "models"
TRAINING_DIR = PIPELINE_DIR / "configs" / "training"
EVAL_DIR = PIPELINE_DIR / "configs" / "evaluation"
DATA_DIR = PIPELINE_DIR / "data" / "uploaded"
OUTPUTS_DIR = PIPELINE_DIR / "outputs"
EVAL_RESULTS_DIR = PIPELINE_DIR / "eval_results"

# Create dirs
for d in [DATA_DIR, OUTPUTS_DIR, EVAL_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_model_configs():
    models = {}
    if MODELS_DIR.exists():
        for f in sorted(MODELS_DIR.glob("*.yaml")):
            try:
                with open(f) as fh:
                    config = yaml.safe_load(fh)
                models[config["model"]["short_name"]] = config
            except Exception:
                pass
    return models

def save_model_config(short_name, config):
    filepath = MODELS_DIR / f"{short_name}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return filepath

def load_training_configs():
    configs = {}
    if TRAINING_DIR.exists():
        for f in sorted(TRAINING_DIR.glob("*.yaml")):
            try:
                with open(f) as fh:
                    configs[f.stem] = yaml.safe_load(fh)
            except Exception:
                pass
    return configs

def load_fintech_prompts():
    prompts_file = EVAL_DIR / "fintech_prompts.json"
    if prompts_file.exists():
        with open(prompts_file) as f:
            return json.load(f)
    return {}

def save_uploaded_file(uploaded_file, subdir=""):
    save_dir = DATA_DIR / subdir
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / uploaded_file.name
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath

def get_model_requirements(config):
    m = config["model"]
    v = config["vram"]
    l = config["lora"]
    t = config["training"]
    r = config["recommended"]

    reqs = {
        "model_name": m["name"],
        "size_billions": m["size_billions"],
        "family": m["family"],
        "context_length": m["context_length"],
        "vram_base_4bit": v.get("qlora_4bit_inference", "N/A"),
        "vram_qlora_training": v.get("qlora_4bit_training", "N/A"),
        "vram_lora_fp16": v.get("lora_fp16_training", "N/A"),
        "lora_rank": l["rank"],
        "lora_alpha": l["alpha"],
        "lora_dropout": l["dropout"],
        "batch_size": t["per_device_train_batch_size"],
        "grad_accum": t["gradient_accumulation_steps"],
        "learning_rate": t["learning_rate"],
        "max_seq_length": t["max_seq_length"],
        "optim": t.get("optim", "adamw_8bit"),
        "notes": r.get("notes", ""),
    }
    return reqs

# Load data
MODEL_CONFIGS = load_model_configs()
TRAINING_CONFIGS = load_training_configs()
FINTECH_PROMPTS = load_fintech_prompts()

# ============================================================
# SESSION STATE INIT
# ============================================================
if "chat_history_before" not in st.session_state:
    st.session_state.chat_history_before = []
if "chat_history_after" not in st.session_state:
    st.session_state.chat_history_after = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = list(MODEL_CONFIGS.keys())[0] if MODEL_CONFIGS else ""
if "training_running" not in st.session_state:
    st.session_state.training_running = False
if "training_results" not in st.session_state:
    st.session_state.training_results = None

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1B3A4B, #2980B9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .model-card {
        background: white;
        padding: 1.2rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #D5D8DC;
        margin-bottom: 0.8rem;
        transition: all 0.2s;
    }
    .model-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        border-color: #2980B9;
    }
    .model-card-selected {
        border: 2px solid #2980B9;
        background: #F0F8FF;
    }
    .req-box {
        background: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2980B9;
        margin: 0.5rem 0;
    }
    .req-box-warn {
        border-left: 4px solid #E67E22;
    }
    .req-box-danger {
        border-left: 4px solid #C0392B;
    }
    .chat-msg-user {
        background: #EBF5FB;
        padding: 0.8rem 1rem;
        border-radius: 1rem;
        margin: 0.3rem 0;
        border: 1px solid #AED6F1;
    }
    .chat-msg-assistant {
        background: #F4F6F7;
        padding: 0.8rem 1rem;
        border-radius: 1rem;
        margin: 0.3rem 0;
        border: 1px solid #D5D8DC;
    }
    .chat-msg-thinking {
        background: #FEF9E7;
        padding: 0.6rem 1rem;
        border-radius: 0.5rem;
        margin: 0.2rem 0;
        border-left: 3px solid #F39C12;
        font-size: 0.9rem;
        color: #7D6608;
    }
    .tag {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 0.2rem;
        margin-bottom: 0.2rem;
    }
    .tag-blue { background: #D4E6F1; color: #1B3A4B; }
    .tag-green { background: #D1F2EB; color: #1E8449; }
    .tag-orange { background: #FDEBD0; color: #D35400; }
    .tag-red { background: #FADBD8; color: #C0392B; }
    .tag-purple { background: #E8DAEF; color: #6C3483; }
    .result-box {
        background: linear-gradient(135deg, #E8F8F5, #D1F2EB);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27AE60;
    }
    div[data-testid="stSidebarNav"] {display: none;}
    .stAlert {padding-top: 0.5rem; padding-bottom: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR - MODEL SELECTOR + NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("## LLM Pipeline")
    st.markdown("---")

    # Model selector - always visible
    st.markdown("### Select Model")
    model_names = list(MODEL_CONFIGS.keys())
    if model_names:
        selected_model = st.selectbox(
            "Model",
            model_names,
            index=0,
            format_func=lambda x: f"{x} ({MODEL_CONFIGS[x]['model']['size_billions']}B)",
            key="sidebar_model",
        )
        st.session_state.selected_model = selected_model
    else:
        st.warning("No models found! Add one from the Model Management page.")

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        [
            "Home",
            "Models & Details",
            "Upload Data",
            "Upload Reasoning Data",
            "Fine-Tuning",
            "Chat with Model",
            "Training Results",
            "Model Settings",
            "Add New Model",
            "Evaluation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption(f"Pipeline v2.0 | {datetime.now().strftime('%Y-%m-%d')}")

# ============================================================
# PAGE: HOME
# ============================================================
if page == "Home":
    st.markdown('<p class="main-header">LLM Fine-Tuning Pipeline</p>', unsafe_allow_html=True)
    st.markdown("**Fine-tune and evaluate open-source LLMs -- for Coding & Fintech with Reasoning**")

    st.markdown("---")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models", len(MODEL_CONFIGS))
    with col2:
        uploaded_count = len(list(DATA_DIR.rglob("*"))) if DATA_DIR.exists() else 0
        st.metric("Uploaded Files", uploaded_count)
    with col3:
        results_count = len(list(OUTPUTS_DIR.rglob("training_stats.json"))) if OUTPUTS_DIR.exists() else 0
        st.metric("Completed Trainings", results_count)
    with col4:
        st.metric("Data Formats", 5)

    st.markdown("---")

    # Quick start guide
    st.subheader("Roadmap")
    steps = [
        ("1", "Select Model", "Models & Details", "Check requirements and pick the right model"),
        ("2", "Upload Data", "Upload Data", "Upload CSV or JSON datasets"),
        ("3", "Upload Reasoning", "Upload Reasoning Data", "Upload CoT and reasoning data"),
        ("4", "Configure", "Fine-Tuning", "Adjust hyperparameters and generate command"),
        ("5", "Chat with Model", "Chat with Model", "Test the model before and after training"),
        ("6", "View Results", "Training Results", "Check loss curves and metrics"),
    ]

    for num, title, nav_page, desc in steps:
        st.markdown(f"**{num}. {title}** -> `{nav_page}` -- _{desc}_")

    st.markdown("---")

    # Top 3 recommendations
    st.subheader("Top 3 Models")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="model-card" style="border-left: 4px solid #8E44AD;">
            <h4>Reasoning & Thinking</h4>
            <b>DeepSeek-R1-Distill-Qwen-7B</b><br>
            <span class="tag tag-purple">Exceptional</span>
            <span class="tag tag-green">16-20 GB</span>
            <p style="font-size:0.85rem; margin-top:0.5rem;">Best for reasoning -- built-in CoT + self-reflection + think tags</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="model-card" style="border-left: 4px solid #2980B9;">
            <h4>Code & Balance</h4>
            <b>Qwen2.5-Coder-7B-Instruct</b><br>
            <span class="tag tag-blue">Strong</span>
            <span class="tag tag-green">16-18 GB</span>
            <p style="font-size:0.85rem; margin-top:0.5rem;">Best 7B for code -- 128K context</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="model-card" style="border-left: 4px solid #E67E22;">
            <h4>Maximum Quality</h4>
            <b>Qwen2.5-Coder-14B-Instruct</b><br>
            <span class="tag tag-orange">Very Strong</span>
            <span class="tag tag-red">24-28 GB</span>
            <p style="font-size:0.85rem; margin-top:0.5rem;">Highest results but tight on VRAM!</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE: MODEL DETAILS
# ============================================================
elif page == "Models & Details":
    st.header("Models & Details")
    st.markdown("Select a model from the sidebar to see its details and requirements")

    st.markdown("---")

    selected = st.session_state.selected_model
    if selected and selected in MODEL_CONFIGS:
        config = MODEL_CONFIGS[selected]
        m = config["model"]
        v = config["vram"]
        l = config["lora"]
        t = config["training"]
        r = config["recommended"]
        reqs = get_model_requirements(config)

        # Model header
        st.markdown(f"""
        <div class="model-card model-card-selected">
            <h2>{m['name']}</h2>
            <span class="tag tag-blue">{m['size_billions']}B</span>
            <span class="tag tag-purple">{m['family']}</span>
            <span class="tag tag-green">{m['type']}</span>
            <span class="tag tag-orange">Context: {m['context_length']:,}</span>
        </div>
        """, unsafe_allow_html=True)

        # Requirements
        st.subheader("Requirements")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="req-box">
                <h4>VRAM</h4>
                <b>Base (4-bit):</b> {v.get('qlora_4bit_inference', 'N/A')}<br>
                <b>QLoRA Training:</b> {v.get('qlora_4bit_training', 'N/A')}<br>
                <b>LoRA FP16 Training:</b> {v.get('lora_fp16_training', 'N/A')}
            </div>
            """, unsafe_allow_html=True)

            vram_str = v.get('qlora_4bit_training', '99+')
            vram_clean = vram_str.replace(" GB", "").replace("~", "")
            try:
                vram_val = float(vram_clean.split("-")[-1])
                if vram_val <= 20:
                    st.success("Fits comfortably on 30GB VRAM")
                elif vram_val <= 28:
                    st.warning("Possible but needs care -- QLoRA 4-bit only")
                else:
                    st.error("VRAM not enough on a 30GB card")
            except ValueError:
                st.error("Will not fit on 30GB VRAM")

        with col2:
            st.markdown(f"""
            <div class="req-box">
                <h4>LoRA Settings</h4>
                <b>Rank:</b> {l['rank']}<br>
                <b>Alpha:</b> {l['alpha']}<br>
                <b>Dropout:</b> {l['dropout']}<br>
                <b>Target Modules:</b> {len(l['target_modules'])} module(s)
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="req-box">
                <h4>Training Settings</h4>
                <b>Batch Size:</b> {t['per_device_train_batch_size']}<br>
                <b>Grad Accum:</b> {t['gradient_accumulation_steps']}<br>
                <b>Effective Batch:</b> {t['per_device_train_batch_size'] * t['gradient_accumulation_steps']}<br>
                <b>LR:</b> {t['learning_rate']}<br>
                <b>Max Seq:</b> {t['max_seq_length']:,}<br>
                <b>Optim:</b> {t.get('optim', 'adamw_8bit')}
            </div>
            """, unsafe_allow_html=True)

        # Notes
        st.markdown("---")
        st.subheader("Notes")
        notes = r.get("notes", "No notes available")
        st.info(notes)

        # HuggingFace link
        hf_name = m["name"]
        st.markdown(f"[View model on HuggingFace](https://huggingface.co/{hf_name})")

        # Chat template
        st.markdown("---")
        st.subheader("Chat Template")
        chat_template = config.get("chat_template", "chatml")
        st.code(chat_template, language="yaml")

        # Full YAML config
        st.markdown("---")
        st.subheader("Full Config File (YAML)")
        yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)
        st.code(yaml_str, language="yaml")

    else:
        st.warning("Select a model from the sidebar!")

# ============================================================
# PAGE: UPLOAD DATA
# ============================================================
elif page == "Upload Data":
    st.header("Upload Data")
    st.markdown("Upload CSV or JSON files for standard fine-tuning (without reasoning)")

    st.markdown("---")

    # Upload section
    st.subheader("Upload Files")

    uploaded_files = st.file_uploader(
        "Upload CSV or JSON files",
        type=["csv", "json", "jsonl"],
        accept_multiple_files=True,
        key="regular_data_upload",
    )

    if uploaded_files:
        for f in uploaded_files:
            filepath = save_uploaded_file(f, "regular")
            st.success(f"{f.name} -- uploaded and saved to `{filepath}`")

            # Preview
            with st.expander(f"Preview {f.name}", expanded=False):
                if f.name.endswith(".csv"):
                    try:
                        df = pd.read_csv(filepath)
                        st.dataframe(df.head(5), use_container_width=True)
                        st.write(f"**Rows:** {len(df):,} | **Columns:** {list(df.columns)}")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
                elif f.name.endswith(".json"):
                    try:
                        with open(filepath) as fh:
                            data = json.load(fh)
                        if isinstance(data, list):
                            st.json(data[:2])
                            st.write(f"**Records:** {len(data):,}")
                        else:
                            st.json(data)
                    except Exception as e:
                        st.error(f"Error: {e}")
                elif f.name.endswith(".jsonl"):
                    try:
                        records = []
                        with open(filepath) as fh:
                            for line in fh:
                                records.append(json.loads(line))
                        st.json(records[:2])
                        st.write(f"**Records:** {len(records):,}")
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown("---")

    # Show already uploaded files
    st.subheader("Uploaded Files")
    regular_dir = DATA_DIR / "regular"
    if regular_dir.exists() and list(regular_dir.glob("*")):
        for f in sorted(regular_dir.glob("*")):
            size_kb = f.stat().st_size / 1024
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f.name)
            with col2:
                st.text(f"{size_kb:.1f} KB")
    else:
        st.info("No files uploaded yet.")

    st.markdown("---")

    # Expected format
    st.subheader("Expected Format")
    example_df = pd.DataFrame({
        "instruction": [
            "Write a Python function to calculate Sharpe ratio",
            "Explain PSD2 SCA requirements",
        ],
        "response": [
            "def sharpe_ratio(returns, risk_free=0.0): ...",
            "PSD2 requires two-factor authentication...",
        ],
        "category": ["financial_analysis", "regulatory_compliance"],
    })
    st.dataframe(example_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Required columns:**
    - `instruction` (required) -- The question or command
    - `response` (required) -- The answer

    **Optional columns:**
    - `system` -- System message
    - `thinking` -- Chain of thought (if you have it)
    - `category` -- Category label
    """)

# ============================================================
# PAGE: UPLOAD REASONING DATA
# ============================================================
elif page == "Upload Reasoning Data":
    st.header("Upload Reasoning Data")
    st.markdown("Upload Chain-of-Thought and reasoning data -- this teaches the model to think before answering")

    st.markdown("---")

    st.markdown("""
    <div class="req-box" style="border-left-color: #8E44AD;">
        <h4>What is Reasoning Data?</h4>
        This is data that contains your <b>thinking chain</b> -- the model learns how to think step-by-step before answering.<br><br>
        <b>Example:</b><br>
        <i>Question:</i> Calculate Sharpe ratio for a portfolio with 12% return, 3% risk-free rate, and 8% deviation<br>
        <i>Thinking:</i> First, compute excess return = 12% - 3% = 9%. Then divide by deviation = 9% / 8% = 1.125<br>
        <i>Answer:</i> Sharpe ratio = 1.125
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Upload reasoning files
    st.subheader("Upload Reasoning Files")

    reasoning_files = st.file_uploader(
        "Upload Reasoning files (CSV or JSON)",
        type=["csv", "json", "jsonl"],
        accept_multiple_files=True,
        key="reasoning_data_upload",
    )

    if reasoning_files:
        for f in reasoning_files:
            filepath = save_uploaded_file(f, "reasoning")
            st.success(f"{f.name} -- uploaded and saved to `{filepath}`")

            with st.expander(f"Preview {f.name}", expanded=False):
                if f.name.endswith(".csv"):
                    try:
                        df = pd.read_csv(filepath)
                        st.dataframe(df.head(5), use_container_width=True)

                        # Check for thinking/reasoning column
                        thinking_cols = ["thinking", "reasoning", "chain_of_thought", "rationale", "explanation"]
                        has_thinking = any(c.lower() in [col.lower() for col in df.columns] for c in thinking_cols)

                        if has_thinking:
                            st.success("Thinking/Reasoning column found!")
                        else:
                            st.warning("No thinking column found -- will be auto-added as CoT scaffolding")
                    except Exception as e:
                        st.error(f"Error: {e}")
                elif f.name.endswith(".json") or f.name.endswith(".jsonl"):
                    try:
                        records = []
                        with open(filepath) as fh:
                            if f.name.endswith(".jsonl"):
                                for line in fh:
                                    records.append(json.loads(line))
                            else:
                                data = json.load(fh)
                                if isinstance(data, list):
                                    records = data
                        st.json(records[:2])
                        st.write(f"**Records:** {len(records):,}")
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown("---")

    # Reasoning format options
    st.subheader("Reasoning Options")

    col1, col2 = st.columns(2)
    with col1:
        add_cot = st.checkbox("Add Chain-of-Thought", value=True, help="Adds step-by-step thinking instructions in system prompt")
        add_thinking_tags = st.checkbox("Add <think/> Tags", value=True, help="Wraps thinking in tags like DeepSeek-R1")
    with col2:
        add_reflection = st.checkbox("Add Self-Reflection", value=False, help="Adds self-review instructions")
        add_tool_use = st.checkbox("Add Tool Use", value=False, help="Adds tool usage examples")

    # Show already uploaded reasoning files
    st.markdown("---")
    st.subheader("Uploaded Reasoning Files")
    reasoning_dir = DATA_DIR / "reasoning"
    if reasoning_dir.exists() and list(reasoning_dir.glob("*")):
        for f in sorted(reasoning_dir.glob("*")):
            size_kb = f.stat().st_size / 1024
            st.text(f"{f.name} ({size_kb:.1f} KB)")
    else:
        st.info("No reasoning files uploaded yet.")

    st.markdown("---")

    # Expected reasoning format
    st.subheader("Expected Reasoning Format")
    example_reasoning = pd.DataFrame({
        "instruction": [
            "Calculate VaR using historical simulation",
            "Design a fraud detection system for payments",
        ],
        "thinking": [
            "Historical VaR sorts past returns and picks the percentile. Step 1: Collect returns. Step 2: Sort ascending. Step 3: Pick the 5th percentile for 95% confidence.",
            "I need multiple layers: real-time scoring, batch analysis, and alert system. Key features: velocity, amount z-score, location mismatch, device fingerprint.",
        ],
        "response": [
            "def historical_var(returns, confidence=0.95): ...",
            "The system architecture includes...",
        ],
        "category": ["risk_assessment", "fraud_detection"],
    })
    st.dataframe(example_reasoning, use_container_width=True, hide_index=True)

# ============================================================
# PAGE: FINE-TUNING
# ============================================================
elif page == "Fine-Tuning":
    st.header("Fine-Tuning")
    st.markdown("Configure training settings and generate the command")

    selected = st.session_state.selected_model
    if not selected or selected not in MODEL_CONFIGS:
        st.warning("Select a model from the sidebar!")
        st.stop()

    config = MODEL_CONFIGS[selected]
    m = config["model"]
    defaults = config["training"]
    lora_defaults = config["lora"]

    # Model info banner
    st.markdown(f"""
    <div class="model-card model-card-selected">
        <h3>{m['name']} ({m['size_billions']}B)</h3>
        <span class="tag tag-blue">{m['family']}</span>
        <span class="tag tag-green">{m['type']}</span>
        <span class="tag tag-orange">Context: {m['context_length']:,}</span>
    </div>
    """, unsafe_allow_html=True)

    # VRAM warning
    vram_str = config["vram"].get('qlora_4bit_training', '99+')
    vram_clean = vram_str.replace(" GB", "").replace("~", "")
    try:
        vram_val = float(vram_clean.split("-")[-1])
        if vram_val > 26:
            st.error(f"High VRAM: {vram_str} -- must use batch_size=1, rank=32, seq=2048")
        elif vram_val > 20:
            st.warning(f"Moderate VRAM: {vram_str} -- use QLoRA 4-bit only")
    except ValueError:
        pass

    st.markdown("---")

    # Training method
    col1, col2 = st.columns(2)
    with col1:
        training_method = st.selectbox(
            "Training Method",
            ["qlora_4bit", "lora_fp16"],
            format_func=lambda x: "QLoRA 4-bit (Recommended)" if x == "qlora_4bit" else "LoRA FP16 (small models only)",
        )
    with col2:
        data_source = st.selectbox(
            "Data Source",
            ["Uploaded Files (regular)", "Reasoning Files", "Custom Path"],
        )

    # Data path based on source
    if data_source == "Uploaded Files (regular)":
        data_path = str(DATA_DIR / "regular")
    elif data_source == "Reasoning Files":
        data_path = str(DATA_DIR / "reasoning")
    else:
        data_path = st.text_input("Data path", value="./data/processed/chatml")

    st.markdown("---")

    # Hyperparameters
    st.subheader("Training Settings")

    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=10, value=defaults["num_train_epochs"])
        lr = st.text_input("Learning Rate", value=str(defaults["learning_rate"]))
        batch_size = st.number_input("Batch Size", min_value=1, max_value=8, value=defaults["per_device_train_batch_size"])
    with col2:
        rank = st.number_input("LoRA Rank", min_value=8, max_value=128, value=lora_defaults["rank"])
        seq_length = st.number_input("Max Seq Length", min_value=512, max_value=8192, value=defaults["max_seq_length"], step=512)
        grad_accum = st.number_input("Gradient Accumulation", min_value=1, max_value=32, value=defaults["gradient_accumulation_steps"])
    with col3:
        warmup = st.number_input("Warmup Ratio", value=float(defaults["warmup_ratio"]), format="%.2f")
        weight_decay = st.number_input("Weight Decay", value=float(defaults["weight_decay"]), format="%.3f")
        max_grad_norm = st.number_input("Max Grad Norm", value=float(defaults["max_grad_norm"]))

    # Reasoning options
    st.markdown("---")
    st.subheader("Reasoning Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        ft_add_cot = st.checkbox("Chain-of-Thought", value=True)
    with col2:
        ft_add_thinking = st.checkbox("<think/> Tags", value=True)
    with col3:
        ft_add_reflection = st.checkbox("Self-Reflection", value=False)

    # Export
    st.markdown("---")
    st.subheader("Export Format")
    export_formats = st.multiselect(
        "Select export formats",
        ["lora", "merged_16bit", "merged_4bit", "gguf"],
        ["lora", "merged_16bit", "gguf"],
        format_func=lambda x: {
            "lora": "LoRA Adapter (~50-200 MB)",
            "merged_16bit": "Merged 16-bit (~14-28 GB)",
            "merged_4bit": "Merged 4-bit (~4-9 GB)",
            "gguf": "GGUF (~3-8 GB, for Ollama)",
        }.get(x, x),
    )

    # Output dir
    output_dir = st.text_input("Output Directory", value=f"./outputs/{selected}")

    # Test run option
    test_run = st.checkbox("Test run (1 epoch, 100 samples only)", value=False)

    # Generate command
    st.markdown("---")
    st.subheader("Training Command")

    cmd_parts = [
        f"python training/finetune.py",
        f"    --model {selected}",
        f"    --data {data_path}",
        f"    --output-dir {output_dir}",
        f"    --training-config {training_method}",
        f"    --epochs {epochs}",
        f"    --lr {lr}",
        f"    --rank {rank}",
        f"    --max-seq-length {seq_length}",
        f"    --batch-size {batch_size}",
        f"    --export {' '.join(export_formats)}",
    ]
    if test_run:
        cmd_parts.append("    --test-run")

    cmd = " \\\n".join(cmd_parts)
    st.code(cmd, language="bash")

    # Effective config summary
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Config Summary:**
        - Model: `{selected}` ({m['size_billions']}B)
        - Method: {training_method}
        - Effective Batch: {batch_size} x {grad_accum} = {batch_size * grad_accum}
        - LoRA: rank={rank}, alpha={rank*2}
        - Epochs: {epochs}, LR: {lr}
        """)
    with col2:
        n_samples = 1000
        steps_per_epoch = max(1, n_samples // (batch_size * grad_accum))
        total_steps = steps_per_epoch * epochs
        st.markdown(f"""
        **Training Estimate:**
        - Steps per epoch: ~{steps_per_epoch}
        - Total steps: ~{total_steps}
        - Time: approximately {total_steps * 2 // 3600}-{total_steps * 5 // 3600} hours*
        - *Varies based on GPU and data size
        """)

    st.info("Copy this command and run it in the terminal. Actual training requires a GPU.")

# ============================================================
# PAGE: CHAT WITH MODEL
# ============================================================
elif page == "Chat with Model":
    st.header("Chat with Model")
    st.markdown("Ask the model before and after training to evaluate the difference")

    selected = st.session_state.selected_model
    if not selected or selected not in MODEL_CONFIGS:
        st.warning("Select a model from the sidebar!")
        st.stop()

    config = MODEL_CONFIGS[selected]
    m = config["model"]

    # Two tabs: Before and After
    tab_before, tab_after = st.tabs(["Before Training (Base)", "After Training (Fine-tuned)"])

    # --- BEFORE TAB ---
    with tab_before:
        st.markdown(f"**Base Model:** `{m['name']}`")

        # Chat history
        for msg in st.session_state.chat_history_before:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg-user"><b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            elif msg["role"] == "thinking":
                st.markdown(f'<div class="chat-msg-thinking"><b>Thinking:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg-assistant"><b>Assistant:</b> {msg["content"]}</div>', unsafe_allow_html=True)

        # Input
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input_before = st.text_input("Question:", key="chat_input_before", label_visibility="collapsed", placeholder="Type your question here...")
        with col2:
            send_before = st.button("Send", key="send_before", use_container_width=True)

        if send_before and user_input_before:
            st.session_state.chat_history_before.append({"role": "user", "content": user_input_before})

            # Try to generate response
            try:
                import torch
                if torch.cuda.is_available():
                    from unsloth import FastLanguageModel
                    if "base_model" not in st.session_state:
                        with st.spinner("Loading model..."):
                            st.session_state.base_model, st.session_state.base_tokenizer = FastLanguageModel.from_pretrained(
                                model_name=m["name"],
                                max_seq_length=2048,
                                load_in_4bit=True,
                            )
                            FastLanguageModel.for_inference(st.session_state.base_model)

                    model_obj = st.session_state.base_model
                    tokenizer_obj = st.session_state.base_tokenizer

                    messages = [{"role": "user", "content": user_input_before}]
                    inputs = tokenizer_obj.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model_obj.device)

                    with torch.no_grad():
                        outputs = model_obj.generate(input_ids=inputs, max_new_tokens=500, temperature=0.7, do_sample=True, pad_token_id=tokenizer_obj.eos_token_id)

                    response = tokenizer_obj.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                    st.session_state.chat_history_before.append({"role": "assistant", "content": response})
                else:
                    st.session_state.chat_history_before.append({"role": "assistant", "content": "No GPU available -- cannot run the model. Try on Colab!"})
            except Exception as e:
                st.session_state.chat_history_before.append({"role": "assistant", "content": f"Error: {str(e)}"})

            st.rerun()

        # Clear button
        if st.button("Clear Chat", key="clear_before"):
            st.session_state.chat_history_before = []
            if "base_model" in st.session_state:
                del st.session_state.base_model
                del st.session_state.base_tokenizer
            st.rerun()

    # --- AFTER TAB ---
    with tab_after:
        ft_model_path = st.text_input("Fine-tuned model path:", value=f"./outputs/{selected}", key="ft_model_path")
        st.markdown(f"**Fine-tuned Model:** `{ft_model_path}`")

        # Chat history
        for msg in st.session_state.chat_history_after:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg-user"><b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            elif msg["role"] == "thinking":
                st.markdown(f'<div class="chat-msg-thinking"><b>Thinking:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg-assistant"><b>Assistant:</b> {msg["content"]}</div>', unsafe_allow_html=True)

        # Input
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input_after = st.text_input("Question:", key="chat_input_after", label_visibility="collapsed", placeholder="Type your question here...")
        with col2:
            send_after = st.button("Send", key="send_after", use_container_width=True)

        if send_after and user_input_after:
            st.session_state.chat_history_after.append({"role": "user", "content": user_input_after})

            try:
                import torch
                if torch.cuda.is_available():
                    from unsloth import FastLanguageModel
                    if "ft_model" not in st.session_state:
                        with st.spinner("Loading fine-tuned model..."):
                            st.session_state.ft_model, st.session_state.ft_tokenizer = FastLanguageModel.from_pretrained(
                                model_name=ft_model_path,
                                max_seq_length=2048,
                                load_in_4bit=True,
                            )
                            FastLanguageModel.for_inference(st.session_state.ft_model)

                    model_obj = st.session_state.ft_model
                    tokenizer_obj = st.session_state.ft_tokenizer

                    messages = [{"role": "user", "content": user_input_after}]
                    inputs = tokenizer_obj.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model_obj.device)

                    with torch.no_grad():
                        outputs = model_obj.generate(input_ids=inputs, max_new_tokens=500, temperature=0.7, do_sample=True, pad_token_id=tokenizer_obj.eos_token_id)

                    response = tokenizer_obj.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                    st.session_state.chat_history_after.append({"role": "assistant", "content": response})
                else:
                    st.session_state.chat_history_after.append({"role": "assistant", "content": "No GPU available -- cannot run the model. Try on Colab!"})
            except Exception as e:
                st.session_state.chat_history_after.append({"role": "assistant", "content": f"Error: {str(e)}"})

            st.rerun()

        if st.button("Clear Chat", key="clear_after"):
            st.session_state.chat_history_after = []
            if "ft_model" in st.session_state:
                del st.session_state.ft_model
                del st.session_state.ft_tokenizer
            st.rerun()

# ============================================================
# PAGE: TRAINING RESULTS
# ============================================================
elif page == "Training Results":
    st.header("Training Results")
    st.markdown("View training results and metrics")

    st.markdown("---")

    # Look for training stats
    stats_files = list(OUTPUTS_DIR.rglob("training_stats.json"))
    eval_files = list(EVAL_RESULTS_DIR.glob("*.json"))

    if stats_files:
        st.subheader("Training Outputs")

        for sf in stats_files:
            model_name = sf.parent.name
            with st.expander(f"Model: {model_name}", expanded=True):
                try:
                    with open(sf) as f:
                        stats = json.load(f)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Final Loss", f"{stats.get('final_loss', 'N/A')}")
                    with col2:
                        st.metric("Epochs", stats.get("epochs", "N/A"))
                    with col3:
                        st.metric("Total Steps", stats.get("total_steps", "N/A"))
                    with col4:
                        st.metric("Training Time", stats.get("training_time", "N/A"))

                    # Show loss curve if available
                    if "loss_history" in stats:
                        loss_data = stats["loss_history"]
                        df_loss = pd.DataFrame(loss_data)
                        st.line_chart(df_loss, x="step", y="loss")

                    # Show config used
                    if "config" in stats:
                        st.json(stats["config"])

                except Exception as e:
                    st.error(f"Error reading stats: {e}")
    else:
        st.info("No training results found yet. Run a training first.")

    st.markdown("---")

    # Evaluation results
    if eval_files:
        st.subheader("Evaluation Results")

        for ef in eval_files:
            model_name = ef.stem
            with st.expander(f"Evaluation: {model_name}", expanded=True):
                try:
                    with open(ef) as f:
                        eval_data = json.load(f)
                    st.json(eval_data)
                except Exception as e:
                    st.error(f"Error reading evaluation: {e}")
    else:
        st.info("No evaluation results found yet.")

# ============================================================
# PAGE: MODEL SETTINGS
# ============================================================
elif page == "Model Settings":
    st.header("Model Settings")
    st.markdown("Edit the configuration and code for the selected model")

    selected = st.session_state.selected_model
    if not selected or selected not in MODEL_CONFIGS:
        st.warning("Select a model from the sidebar!")
        st.stop()

    config = MODEL_CONFIGS[selected]

    # Edit YAML config
    st.subheader("YAML Configuration")
    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)
    edited_yaml = st.text_area("Edit config (YAML)", value=yaml_str, height=400, key="edit_yaml")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save YAML Config", key="save_yaml_btn"):
            try:
                new_config = yaml.safe_load(edited_yaml)
                save_model_config(selected, new_config)
                st.success("Config saved! Refresh to see changes.")
            except Exception as e:
                st.error(f"Invalid YAML: {e}")
    with col2:
        if st.button("Reset to Default", key="reset_yaml_btn"):
            st.rerun()

    st.markdown("---")

    # Edit model-specific code
    st.subheader("Model-Specific Code")
    st.markdown("Edit the Python code that runs when fine-tuning this specific model")

    default_code = f'''# Fine-tuning code for {selected}
# This code is executed when you run fine-tuning for this model

from unsloth import FastLanguageModel
import torch

def load_model():
    """Load the model with optimal settings for {selected}"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="{config["model"]["name"]}",
        max_seq_length={config["training"]["max_seq_length"]},
        load_in_4bit=True,
        dtype=None,
    )
    return model, tokenizer

def setup_lora(model):
    """Configure LoRA adapters for this model"""
    model = FastLanguageModel.get_peft_model(
        model,
        r={config["lora"]["rank"]},
        lora_alpha={config["lora"]["alpha"]},
        lora_dropout={config["lora"]["dropout"]},
        target_modules={config["lora"]["target_modules"]},
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model

def train(model, tokenizer, dataset):
    """Run the training loop"""
    from trl import SFTTrainer
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        per_device_train_batch_size={config["training"]["per_device_train_batch_size"]},
        gradient_accumulation_steps={config["training"]["gradient_accumulation_steps"]},
        warmup_ratio={config["training"]["warmup_ratio"]},
        num_train_epochs={config["training"]["num_train_epochs"]},
        learning_rate={config["training"]["learning_rate"]},
        max_grad_norm={config["training"]["max_grad_norm"]},
        weight_decay={config["training"]["weight_decay"]},
        logging_steps=10,
        optim="{config["training"].get("optim", "adamw_8bit")}",
        seed=3407,
        output_dir="./outputs/{selected}",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    return trainer
'''

    edited_code = st.text_area("Edit model code (Python)", value=default_code, height=500, key="edit_code")

    if st.button("Save Model Code", key="save_code_btn"):
        code_file = MODELS_DIR / f"{selected}_code.py"
        with open(code_file, "w") as f:
            f.write(edited_code)
        st.success(f"Code saved to {code_file}")

# ============================================================
# PAGE: ADD NEW MODEL
# ============================================================
elif page == "Add New Model":
    st.header("Add New Model")
    st.markdown("Add a new model to the pipeline by filling in its configuration")

    st.markdown("---")

    # Basic info
    st.subheader("Model Information")
    col1, col2 = st.columns(2)
    with col1:
        new_short_name = st.text_input("Short Name (no spaces)", value="my-model-7b", key="new_short_name")
        new_model_name = st.text_input("HuggingFace Model ID", value="organization/model-name", key="new_model_name")
        new_size = st.number_input("Size (Billions)", value=7.0, step=0.5, key="new_size")
    with col2:
        new_family = st.selectbox("Family", ["qwen2", "llama3", "mistral", "phi", "deepseek", "other"], key="new_family")
        new_type = st.selectbox("Type", ["instruct", "base", "chat"], key="new_type")
        new_context = st.number_input("Context Length", value=8192, step=1024, key="new_context")

    # Chat template
    new_chat_template = st.selectbox("Chat Template", ["chatml", "llama3", "alpaca", "deepseek-r1", "sharegpt"], key="new_chat_template")

    # VRAM estimates
    st.markdown("---")
    st.subheader("VRAM Estimates")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_vram_inference = st.text_input("4-bit Inference VRAM", value="~6 GB", key="new_vram_inf")
    with col2:
        new_vram_qlora = st.text_input("QLoRA 4-bit Training VRAM", value="~16 GB", key="new_vram_qlora")
    with col3:
        new_vram_lora = st.text_input("LoRA FP16 Training VRAM", value="~28 GB", key="new_vram_lora")

    # LoRA defaults
    st.markdown("---")
    st.subheader("LoRA Defaults")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_rank = st.number_input("LoRA Rank", value=32, step=8, key="new_rank")
        new_alpha = st.number_input("LoRA Alpha", value=64, step=8, key="new_alpha")
    with col2:
        new_dropout = st.number_input("LoRA Dropout", value=0.0, step=0.05, key="new_dropout")
    with col3:
        new_target_modules = st.text_input("Target Modules (comma-separated)", value="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", key="new_target_modules")

    # Training defaults
    st.markdown("---")
    st.subheader("Training Defaults")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_batch = st.number_input("Batch Size", value=2, key="new_batch")
        new_grad = st.number_input("Gradient Accumulation", value=4, key="new_grad")
        new_epochs = st.number_input("Epochs", value=3, key="new_epochs_def")
    with col2:
        new_lr = st.text_input("Learning Rate", value="2e-4", key="new_lr")
        new_seq = st.number_input("Max Seq Length", value=2048, step=512, key="new_seq")
        new_warmup = st.number_input("Warmup Ratio", value=0.05, step=0.01, key="new_warmup")
    with col3:
        new_weight_decay = st.number_input("Weight Decay", value=0.01, step=0.005, key="new_weight_decay")
        new_max_grad = st.number_input("Max Grad Norm", value=1.0, step=0.1, key="new_max_grad")
        new_optim = st.selectbox("Optimizer", ["adamw_8bit", "adamw_torch", "paged_adamw_8bit"], key="new_optim")

    # Notes
    st.markdown("---")
    st.subheader("Notes")
    new_notes = st.text_area("Notes about this model", value="", key="new_notes")

    # Build and save
    st.markdown("---")
    if st.button("Add Model", key="add_model_btn"):
        new_config = {
            "model": {
                "short_name": new_short_name,
                "name": new_model_name,
                "size_billions": new_size,
                "family": new_family,
                "type": new_type,
                "context_length": new_context,
            },
            "chat_template": new_chat_template,
            "vram": {
                "qlora_4bit_inference": new_vram_inference,
                "qlora_4bit_training": new_vram_qlora,
                "lora_fp16_training": new_vram_lora,
            },
            "lora": {
                "rank": new_rank,
                "alpha": new_alpha,
                "dropout": new_dropout,
                "target_modules": [m.strip() for m in new_target_modules.split(",")],
            },
            "training": {
                "per_device_train_batch_size": new_batch,
                "gradient_accumulation_steps": new_grad,
                "num_train_epochs": new_epochs,
                "learning_rate": new_lr,
                "max_seq_length": new_seq,
                "warmup_ratio": new_warmup,
                "weight_decay": new_weight_decay,
                "max_grad_norm": new_max_grad,
                "optim": new_optim,
            },
            "recommended": {
                "notes": new_notes,
            },
        }

        filepath = save_model_config(new_short_name, new_config)
        st.success(f"Model added! Config saved to `{filepath}`. Refresh the page to see it in the sidebar.")

# ============================================================
# PAGE: EVALUATION
# ============================================================
elif page == "Evaluation":
    st.header("Evaluation")
    st.markdown("Run evaluation benchmarks on fine-tuned models")

    st.markdown("---")

    # Model selection for evaluation
    selected = st.session_state.selected_model
    if not selected or selected not in MODEL_CONFIGS:
        st.warning("Select a model from the sidebar!")
        st.stop()

    config = MODEL_CONFIGS[selected]
    m = config["model"]

    st.markdown(f"**Evaluating:** `{m['name']}` ({m['size_billions']}B)")

    st.markdown("---")

    # Evaluation options
    st.subheader("Evaluation Benchmarks")

    benchmarks = st.multiselect(
        "Select benchmarks to run",
        [
            "perplexity",
            "bleu",
            "rouge",
            "exact_match",
            "f1_score",
            "gpt_as_judge",
            "fintech_custom",
        ],
        ["perplexity", "bleu", "rouge", "exact_match"],
        format_func=lambda x: {
            "perplexity": "Perplexity (lower is better)",
            "bleu": "BLEU Score (higher is better)",
            "rouge": "ROUGE Score (higher is better)",
            "exact_match": "Exact Match (higher is better)",
            "f1_score": "F1 Score (higher is better)",
            "gpt_as_judge": "GPT-as-Judge (comparative eval)",
            "fintech_custom": "Custom Fintech Benchmark",
        }.get(x, x),
    )

    st.markdown("---")

    # Model path
    ft_model_path = st.text_input("Fine-tuned model path:", value=f"./outputs/{selected}", key="eval_model_path")

    # Test prompts
    st.subheader("Test Prompts")
    st.markdown("Select from built-in fintech prompts or write your own")

    use_builtin = st.checkbox("Use built-in fintech prompts", value=True)
    custom_prompt = st.text_area("Custom test prompt:", value="", key="custom_eval_prompt")

    # Generate eval command
    st.markdown("---")
    st.subheader("Evaluation Command")

    eval_cmd_parts = [
        f"python evaluation/evaluate.py",
        f"    --model-path {ft_model_path}",
        f"    --benchmarks {' '.join(benchmarks)}",
    ]
    if use_builtin:
        eval_cmd_parts.append("    --use-fintech-prompts")
    if custom_prompt:
        eval_cmd_parts.append(f'    --custom-prompt "{custom_prompt}"')

    eval_cmd = " \\\n".join(eval_cmd_parts)
    st.code(eval_cmd, language="bash")

    st.info("Copy this command and run it in the terminal. Evaluation requires a GPU and the fine-tuned model to be available.")

    # Show existing results
    st.markdown("---")
    st.subheader("Existing Evaluation Results")

    eval_files = list(EVAL_RESULTS_DIR.glob("*.json"))
    if eval_files:
        for ef in sorted(eval_files):
            with st.expander(ef.stem, expanded=False):
                try:
                    with open(ef) as f:
                        eval_data = json.load(f)
                    st.json(eval_data)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("No evaluation results found yet.")
