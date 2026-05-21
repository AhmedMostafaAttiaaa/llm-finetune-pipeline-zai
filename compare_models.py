"""
Model Comparison Script
========================
Compares fine-tuned models across benchmarks and generates recommendations.

Usage:
    python compare_models.py --results-dir ./eval_results --output comparison_report.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL COMPARISON LOGIC
# ============================================================================

# Default model characteristics for recommendation engine
MODEL_PROFILES = {
    "qwen2.5-coder-1.5b": {
        "name": "Qwen2.5-Coder-1.5B-Instruct",
        "size_billions": 1.5,
        "strengths": ["coding", "efficiency", "speed"],
        "weaknesses": ["complex reasoning", "long context"],
        "vram_training_gb": "6-8",
        "recommended_for": ["prototyping", "edge deployment", "fast iteration", "limited VRAM"],
        "reasoning_tier": "basic",
    },
    "phi-2": {
        "name": "Phi-2 (2.7B)",
        "size_billions": 2.7,
        "strengths": ["reasoning", "math", "logic"],
        "weaknesses": ["short context (2048)", "base model (needs instruction tuning)"],
        "vram_training_gb": "8-10",
        "recommended_for": ["reasoning tasks", "math problems", "quick experiments"],
        "reasoning_tier": "moderate",
    },
    "qwen2.5-coder-7b": {
        "name": "Qwen2.5-Coder-7B-Instruct",
        "size_billions": 7,
        "strengths": ["coding", "long context", "instruction following", "fintech reasoning"],
        "weaknesses": ["moderate VRAM required"],
        "vram_training_gb": "16-18",
        "recommended_for": ["coding + fintech", "production deployment", "balanced performance"],
        "reasoning_tier": "strong",
    },
    "llama-3.1-8b": {
        "name": "Llama-3.1-8B-Instruct",
        "size_billions": 8,
        "strengths": ["general purpose", "reasoning", "instruction following", "community support"],
        "weaknesses": ["tight VRAM fit", "less coding-specific"],
        "vram_training_gb": "18-20",
        "recommended_for": ["general fintech", "conversational AI", "well-rounded tasks"],
        "reasoning_tier": "strong",
    },
    "deepseek-coder-7b": {
        "name": "DeepSeek-Coder-V2-Lite (7B MoE)",
        "size_billions": 7,
        "strengths": ["coding", "MoE efficiency", "long context"],
        "weaknesses": ["MoE complexity", "trust_remote_code required"],
        "vram_training_gb": "16-18",
        "recommended_for": ["coding-heavy tasks", "efficient inference", "fintech code generation"],
        "reasoning_tier": "strong",
    },
    "deepseek-r1-qwen-7b": {
        "name": "DeepSeek-R1-Distill-Qwen-7B",
        "size_billions": 7,
        "strengths": ["chain-of-thought", "self-reflection", "multi-step reasoning", "thinking"],
        "weaknesses": ["verbose outputs", "can overthink simple questions"],
        "vram_training_gb": "16-20",
        "recommended_for": ["REASONING & THINKING (primary goal)", "complex fintech analysis", "self-correction"],
        "reasoning_tier": "exceptional",
    },
    "mistral-7b": {
        "name": "Mistral-7B-Instruct-v0.3",
        "size_billions": 7,
        "strengths": ["efficiency", "speed", "balanced performance", "battle-tested"],
        "weaknesses": ["less specialized than coding models"],
        "vram_training_gb": "16-18",
        "recommended_for": ["production systems", "fast inference", "balanced coding + fintech"],
        "reasoning_tier": "strong",
    },
    "qwen2.5-coder-14b": {
        "name": "Qwen2.5-Coder-14B-Instruct",
        "size_billions": 14,
        "strengths": ["best coding performance", "highest quality outputs", "long context"],
        "weaknesses": ["TIGHT VRAM (30GB limit)", "slow training", "limited batch size"],
        "vram_training_gb": "24-28",
        "recommended_for": ["maximum code quality", "when VRAM allows", "complex coding tasks"],
        "reasoning_tier": "very strong",
    },
}


def load_eval_results(results_dir: str) -> List[Dict]:
    """Load all evaluation results from a directory."""
    results = []
    results_path = Path(results_dir)
    
    if results_path.is_file() and results_path.suffix == ".json":
        with open(results_path) as f:
            results.append(json.load(f))
    elif results_path.is_dir():
        for f in sorted(results_path.glob("**/eval_results_*.json")):
            with open(f) as f:
                results.append(json.load(f))
        # Also check training_stats files
        for f in sorted(results_path.glob("**/training_stats.json")):
            with open(f) as fh:
                results.append({"training_stats": json.load(fh)})
    
    return results


def compare_benchmarks(results: List[Dict]) -> Dict:
    """Compare models across benchmarks."""
    comparison = {}
    
    for result in results:
        model_name = result.get("model", "unknown")
        benchmarks = result.get("benchmarks", {})
        
        for bench_name, bench_data in benchmarks.items():
            if bench_name not in comparison:
                comparison[bench_name] = {}
            
            # Extract key metrics
            metrics = {}
            for key, value in bench_data.items():
                if isinstance(value, (int, float)) and key not in ["num_fewshot", "num_questions", "total", "correct"]:
                    metrics[key] = value
            
            comparison[bench_name][model_name] = metrics
    
    return comparison


def generate_recommendations(comparison: Dict, priority: str = "reasoning") -> Dict:
    """Generate model recommendations based on evaluation results and priorities."""
    recommendations = {
        "priority": priority,
        "top_picks": [],
        "detailed_analysis": {},
    }
    
    # Score each model
    model_scores = {}
    for model_key, profile in MODEL_PROFILES.items():
        score = 0
        analysis = {"strengths": [], "concerns": [], "fit_score": 0}
        
        # Priority-based scoring
        if priority == "reasoning":
            # Reward reasoning tier
            tier_scores = {"basic": 1, "moderate": 2, "strong": 3, "very strong": 4, "exceptional": 5}
            score += tier_scores.get(profile["reasoning_tier"], 0) * 3
            if "reasoning" in profile["strengths"]:
                score += 2
            if "chain-of-thought" in profile["strengths"]:
                score += 3
            if "self-reflection" in profile["strengths"]:
                score += 3
            if "thinking" in profile["strengths"]:
                score += 3
        
        if priority == "coding":
            if "coding" in profile["strengths"]:
                score += 3
            if "code generation" in profile["strengths"]:
                score += 2
        
        if priority == "fintech":
            if "fintech" in str(profile["strengths"]):
                score += 3
            if "financial" in str(profile["strengths"]):
                score += 2
        
        # VRAM safety margin
        vram_str = profile["vram_training_gb"]
        vram_max = int(vram_str.split("-")[-1]) if "-" in vram_str else int(vram_str)
        if vram_max <= 20:
            score += 2  # Good safety margin on 30GB
            analysis["strengths"].append("Comfortable VRAM headroom")
        elif vram_max <= 28:
            score += 1  # Tight but workable
            analysis["concerns"].append("Tight VRAM - monitor carefully")
        else:
            analysis["concerns"].append("May OOM on 30GB - use minimal batch size")
        
        # Size efficiency
        if profile["size_billions"] <= 3:
            score += 1  # Fast iteration bonus
        
        analysis["fit_score"] = score
        model_scores[model_key] = score
        recommendations["detailed_analysis"][model_key] = analysis
    
    # Sort and pick top models
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    recommendations["top_picks"] = [
        {
            "rank": i + 1,
            "model": model_key,
            "name": MODEL_PROFILES[model_key]["name"],
            "score": score,
            "reason": get_recommendation_reason(model_key, priority),
        }
        for i, (model_key, score) in enumerate(sorted_models[:5])
    ]
    
    return recommendations


def get_recommendation_reason(model_key: str, priority: str) -> str:
    """Generate a human-readable recommendation reason."""
    profile = MODEL_PROFILES.get(model_key, {})
    reasons = []
    
    if priority == "reasoning":
        if profile.get("reasoning_tier") == "exceptional":
            reasons.append("Best reasoning capabilities with built-in chain-of-thought and self-reflection")
        elif profile.get("reasoning_tier") == "very strong":
            reasons.append("Very strong reasoning with large model capacity")
        elif profile.get("reasoning_tier") == "strong":
            reasons.append("Strong reasoning with good efficiency")
    
    if "coding" in profile.get("strengths", []):
        reasons.append("Excellent coding abilities")
    if "MoE efficiency" in profile.get("strengths", []):
        reasons.append("Efficient MoE architecture for fast inference")
    
    vram = profile.get("vram_training_gb", "")
    if "6-8" in vram or "8-10" in vram:
        reasons.append("Low VRAM requirement allows for larger batch sizes and faster training")
    elif "24-28" in vram:
        reasons.append("Requires careful VRAM management on 30GB GPU")
    
    return "; ".join(reasons) if reasons else "General-purpose model"


def generate_comparison_table(comparison: Dict) -> str:
    """Generate a markdown comparison table."""
    lines = []
    lines.append("| Benchmark | " + " | ".join(MODEL_PROFILES.keys()) + " |")
    lines.append("|" + "|".join(["---"] * (len(MODEL_PROFILES) + 1)) + "|")
    
    for bench_name, models in comparison.items():
        row = [bench_name]
        for model_key in MODEL_PROFILES:
            if model_key in models:
                metrics = models[model_key]
                # Format metrics
                metric_str = ", ".join(f"{k}: {v}" for k, v in metrics.items() if isinstance(v, (int, float)))
                row.append(metric_str if metric_str else "N/A")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare fine-tuned models and generate recommendations")
    parser.add_argument("--results-dir", required=True, help="Directory with evaluation results")
    parser.add_argument("--output", default="./model_comparison.json", help="Output file")
    parser.add_argument("--priority", default="reasoning", 
                        choices=["reasoning", "coding", "fintech", "balanced"],
                        help="Optimization priority")
    parser.add_argument("--format", default="json", choices=["json", "markdown"],
                        help="Output format")
    args = parser.parse_args()
    
    # Load results
    results = load_eval_results(args.results_dir)
    
    if not results:
        logger.warning("No evaluation results found. Generating recommendations based on model profiles only.")
        comparison = {}
    else:
        comparison = compare_benchmarks(results)
    
    # Generate recommendations
    recommendations = generate_recommendations(comparison, args.priority)
    
    # Generate output
    output = {
        "timestamp": datetime.now().isoformat(),
        "priority": args.priority,
        "comparison": comparison,
        "recommendations": recommendations,
        "model_profiles": MODEL_PROFILES,
    }
    
    # Save
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Comparison saved to {args.output}")
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"MODEL COMPARISON (Priority: {args.priority})")
    print("=" * 70)
    
    print("\n🏆 TOP PICKS:")
    for pick in recommendations["top_picks"]:
        print(f"\n  #{pick['rank']}: {pick['name']} (Score: {pick['score']})")
        print(f"     Reason: {pick['reason']}")
    
    if comparison:
        print("\n📊 BENCHMARK COMPARISON:")
        print(generate_comparison_table(comparison))
    
    print(f"\nFull results: {args.output}")


if __name__ == "__main__":
    main()
