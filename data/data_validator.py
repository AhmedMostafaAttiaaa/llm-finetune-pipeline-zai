"""
Data Validator Script
=====================
Validates your CSV dataset before fine-tuning.
Reports quality metrics, potential issues, and recommendations.

Usage:
    python data_validator.py --input data.csv
"""

import argparse
import json
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COLUMN_ALIASES = {
    "instruction": ["instruction", "user", "human", "question", "prompt", "input", "query"],
    "response": ["response", "assistant", "answer", "output", "reply", "completion"],
    "system": ["system", "context", "background", "system_prompt"],
    "thinking": ["thinking", "reasoning", "explanation", "chain_of_thought", "rationale"],
    "category": ["category", "domain", "type", "topic", "label"],
}


def detect_columns(df):
    detected = {}
    columns_lower = {col.lower().strip(): col for col in df.columns}
    for role, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in columns_lower:
                detected[role] = columns_lower[alias]
                break
    return detected


def validate_csv(filepath: str) -> dict:
    """Comprehensive CSV validation."""
    results = {"file": filepath, "issues": [], "warnings": [], "recommendations": [], "stats": {}}
    
    # 1. Load file
    try:
        df = pd.read_csv(filepath)
        results["stats"]["total_rows"] = len(df)
        results["stats"]["total_columns"] = len(df.columns)
        results["stats"]["columns"] = list(df.columns)
    except Exception as e:
        results["issues"].append(f"Cannot load CSV: {e}")
        return results
    
    # 2. Column detection
    detected = detect_columns(df)
    results["detected_columns"] = detected
    
    if "instruction" not in detected:
        results["issues"].append("No instruction/user column detected. Fine-tuning requires user prompts.")
    if "response" not in detected:
        results["issues"].append("No response/assistant column detected. Fine-tuning requires model responses.")
    
    if not detected.get("thinking"):
        results["recommendations"].append(
            "No thinking/reasoning column found. Add a 'thinking' or 'chain_of_thought' column "
            "to enable reasoning fine-tuning (CoT, self-reflection)."
        )
    if not detected.get("system"):
        results["recommendations"].append(
            "No system column found. A system prompt helps set model behavior. "
            "Consider adding a 'system' column with domain context."
        )
    
    # 3. Data quality checks
    for role, col in detected.items():
        col_data = df[col]
        
        # Null values
        null_count = col_data.isna().sum()
        if null_count > 0:
            results["warnings"].append(f"'{col}' ({role}): {null_count} null values ({null_count/len(df)*100:.1f}%)")
        
        # Empty strings
        empty_count = (col_data.astype(str).str.strip() == "").sum()
        if empty_count > 0:
            results["warnings"].append(f"'{col}' ({role}): {empty_count} empty strings")
        
        # Length distribution
        lengths = col_data.dropna().astype(str).str.len()
        results["stats"][f"{role}_length"] = {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "p95": float(lengths.quantile(0.95)),
            "p99": float(lengths.quantile(0.99)),
        }
        
        # Too short (< 10 chars)
        short_count = (lengths < 10).sum()
        if short_count > 0 and role in ["instruction", "response"]:
            results["warnings"].append(f"'{col}' ({role}): {short_count} very short entries (<10 chars)")
        
        # Too long (> 32K chars)
        long_count = (lengths > 32000).sum()
        if long_count > 0:
            results["warnings"].append(f"'{col}' ({role}): {long_count} very long entries (>32K chars)")
    
    # 4. Duplicate check
    if "instruction" in detected:
        dup_count = df[detected["instruction"]].duplicated().sum()
        if dup_count > 0:
            results["warnings"].append(f"{dup_count} duplicate instructions found ({dup_count/len(df)*100:.1f}%)")
    
    # 5. Category distribution
    if "category" in detected:
        cat_counts = df[detected["category"]].value_counts().to_dict()
        results["stats"]["category_distribution"] = cat_counts
        
        # Check balance
        if len(cat_counts) > 1:
            max_cat = max(cat_counts.values())
            min_cat = min(cat_counts.values())
            if max_cat / min_cat > 10:
                results["warnings"].append(
                    f"Category imbalance: max={max_cat}, min={min_cat}. "
                    f"Consider resampling for balanced training."
                )
    
    # 6. Dataset size assessment
    n = len(df)
    if n < 100:
        results["issues"].append(
            f"Dataset is very small ({n} samples). Minimum recommended: 500 samples. "
            f"Consider data augmentation or collecting more data."
        )
    elif n < 500:
        results["warnings"].append(
            f"Dataset is small ({n} samples). Recommended: 1000+ for LoRA, 5000+ for full fine-tuning."
        )
    elif n < 2000:
        results["recommendations"].append(
            f"Dataset size ({n}) is adequate for LoRA fine-tuning. "
            f"For better results, aim for 5000+ samples."
        )
    else:
        results["recommendations"].append(f"Dataset size ({n}) is good for LoRA fine-tuning.")
    
    # 7. Code content detection
    if "instruction" in detected:
        code_keywords = ["def ", "class ", "import ", "function", "return ", "print(", "```", "// ", "/* ", "#include"]
        code_rows = 0
        for kw in code_keywords:
            code_rows += df[detected["instruction"]].astype(str).str.contains(kw, regex=False, na=False).sum()
        code_pct = min(code_rows / n * 100, 100)
        results["stats"]["code_content_pct"] = round(code_pct, 1)
        if code_pct > 10:
            results["recommendations"].append(
                f"High code content detected ({code_pct:.0f}%). "
                f"Consider using a code-specialized model like Qwen2.5-Coder or DeepSeek-Coder."
            )
    
    # 8. Finance content detection
    if "instruction" in detected or "response" in detected:
        fin_keywords = ["stock", "portfolio", "risk", "trading", "financial", "revenue", "profit",
                        "market", "investment", "compliance", "regulation", "bank", "loan", "interest rate"]
        text_col = detected.get("response", detected.get("instruction", ""))
        if text_col:
            fin_rows = sum(df[text_col].astype(str).str.lower().str.contains(kw, regex=False, na=False).sum()
                          for kw in fin_keywords)
            fin_pct = min(fin_rows / n * 100, 100)
            results["stats"]["fintech_content_pct"] = round(fin_pct, 1)
            if fin_pct > 10:
                results["recommendations"].append(
                    f"Strong fintech content detected ({fin_pct:.0f}%). "
                    f"Consider adding FinQA benchmark to evaluation."
                )
    
    # Overall quality score
    issue_penalty = len(results["issues"]) * 20
    warning_penalty = len(results["warnings"]) * 5
    score = max(0, 100 - issue_penalty - warning_penalty)
    results["quality_score"] = score
    
    if score >= 80:
        results["quality_label"] = "GOOD - Ready for fine-tuning with minor fixes"
    elif score >= 60:
        results["quality_label"] = "FAIR - Address warnings before fine-tuning"
    elif score >= 40:
        results["quality_label"] = "POOR - Significant issues need resolution"
    else:
        results["quality_label"] = "CRITICAL - Dataset not suitable for fine-tuning"
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate CSV dataset for LLM fine-tuning")
    parser.add_argument("--input", required=True, help="Path to CSV file")
    parser.add_argument("--output", default=None, help="Save validation report to JSON")
    args = parser.parse_args()
    
    results = validate_csv(args.input)
    
    # Print report
    print("\n" + "=" * 70)
    print("DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"\nFile: {results['file']}")
    print(f"Total Rows: {results['stats'].get('total_rows', 'N/A')}")
    print(f"Total Columns: {results['stats'].get('total_columns', 'N/A')}")
    print(f"Detected Columns: {results.get('detected_columns', {})}")
    
    print(f"\nQuality Score: {results['quality_score']}/100 - {results['quality_label']}")
    
    if results["issues"]:
        print(f"\n🚨 ISSUES ({len(results['issues'])}):")
        for issue in results["issues"]:
            print(f"  ❌ {issue}")
    
    if results["warnings"]:
        print(f"\n⚠️ WARNINGS ({len(results['warnings'])}):")
        for warning in results["warnings"]:
            print(f"  ⚠ {warning}")
    
    if results["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS ({len(results['recommendations'])}):")
        for rec in results["recommendations"]:
            print(f"  💡 {rec}")
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
