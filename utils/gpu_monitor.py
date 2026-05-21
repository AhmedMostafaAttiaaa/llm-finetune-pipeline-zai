# GPU Monitor Utility
# Monitors VRAM usage during training

import torch
import time
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def get_gpu_stats() -> dict:
    """Get current GPU statistics."""
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    stats = {
        "gpu_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "total_vram_gb": round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 2),
        "allocated_vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "reserved_vram_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        "free_vram_gb": round(
            (torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated()) / 1024**3, 2
        ),
        "utilization_pct": round(
            torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_mem * 100, 1
        ),
    }
    return stats


def monitor_gpu(interval_seconds: int = 60, log_file: str = "gpu_monitor.jsonl", 
                max_duration_hours: float = 24.0):
    """Monitor GPU usage and log to file."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    max_iterations = int(max_duration_hours * 3600 / interval_seconds)
    
    logger.info(f"Starting GPU monitor (interval: {interval_seconds}s, log: {log_file})")
    
    for i in range(max_iterations):
        stats = get_gpu_stats()
        stats["timestamp"] = datetime.now().isoformat()
        
        with open(log_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
        
        if stats.get("utilization_pct", 0) > 95:
            logger.warning(f"⚠️ High VRAM usage: {stats['utilization_pct']}%")
        
        time.sleep(interval_seconds)


def check_vram_for_model(model_size_billions: float, quantization: str = "4bit") -> dict:
    """Check if the model can fit in available VRAM."""
    stats = get_gpu_stats()
    if not stats["gpu_available"]:
        return {"can_fit": False, "reason": "No GPU available"}
    
    total_vram = stats["total_vram_gb"]
    
    # Rough estimates
    if quantization == "4bit":
        model_vram = model_size_billions * 0.6  # ~0.6 GB per billion params in 4-bit
        training_overhead = model_size_billions * 1.2  # LoRA + optimizer + activations
    else:
        model_vram = model_size_billions * 2.0  # FP16
        training_overhead = model_size_billions * 2.0
    
    total_needed = model_vram + training_overhead
    can_fit = total_needed <= total_vram * 0.9  # Leave 10% headroom
    
    return {
        "can_fit": can_fit,
        "total_vram_gb": total_vram,
        "estimated_needed_gb": round(total_needed, 1),
        "headroom_gb": round(total_vram - total_needed, 1),
        "recommendation": "OK" if can_fit else "Reduce batch size, seq length, or LoRA rank",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, help="Monitor interval in seconds")
    parser.add_argument("--log-file", default="gpu_monitor.jsonl")
    parser.add_argument("--check-model", type=float, default=None, help="Check if model size (in B) fits in VRAM")
    args = parser.parse_args()
    
    if args.check_model:
        result = check_vram_for_model(args.check_model)
        print(json.dumps(result, indent=2))
    else:
        monitor_gpu(args.interval, args.log_file)
