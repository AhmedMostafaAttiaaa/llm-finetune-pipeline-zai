"""
Embedding Model Fine-Tuning Pipeline
======================================
Fine-tune and evaluate NLP embedding models for fintech/coding domain.

Supports:
  - BAAI/bge-large-en-v1.5
  - intfloat/e5-large-v2
  - BAAI/bge-m3
  - thenlper/gte-large
  - sentence-transformers/all-MiniLM-L6-v2 (baseline)

Usage:
    # Fine-tune an embedding model
    python embedding_pipeline.py --model BAAI/bge-large-en-v1.5 --data ./data/processed --train

    # Evaluate embedding quality
    python embedding_pipeline.py --model BAAI/bge-large-en-v1.5 --data ./data/processed --evaluate

    # Compare embedding models
    python embedding_pipeline.py --compare --models BAAI/bge-large-en-v1.5 intfloat/e5-large-v2
"""

import argparse
import json
import os
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Supported embedding models with VRAM estimates
EMBEDDING_MODELS = {
    "bge-large": {
        "name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "max_length": 512,
        "vram_gb": "~2 GB",
        "description": "Best overall English embedding model. Strong for semantic search and retrieval.",
    },
    "e5-large": {
        "name": "intfloat/e5-large-v2",
        "dimension": 1024,
        "max_length": 512,
        "vram_gb": "~2 GB",
        "description": "Excellent for text similarity and clustering. Good for document retrieval.",
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "dimension": 1024,
        "max_length": 8192,
        "vram_gb": "~3 GB",
        "description": "Multilingual, multi-function (dense + sparse + ColBERT). Long context support.",
    },
    "gte-large": {
        "name": "thenlper/gte-large",
        "dimension": 1024,
        "max_length": 512,
        "vram_gb": "~2 GB",
        "description": "General text embedding. Good for classification and search.",
    },
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "max_length": 256,
        "vram_gb": "~0.5 GB",
        "description": "Lightweight baseline. Fast but lower quality. Good for prototyping.",
    },
}


def finetune_embedding_model(
    model_name: str,
    train_data_path: str,
    output_dir: str = "./embedding_outputs",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
):
    """Fine-tune an embedding model using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
        raise
    
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Load training data
    # Expected format: JSONL with pairs (query, positive, [negative])
    import json
    train_examples = []
    
    with open(train_data_path) as f:
        for line in f:
            item = json.loads(line)
            if "positive" in item:
                # Pair/triplet format
                train_examples.append(
                    InputExample(texts=[item["query"], item["positive"]], label=item.get("score", 1.0))
                )
            elif "messages" in item:
                # Chat format - use instruction/response as pairs
                messages = item["messages"]
                query = next((m["content"] for m in messages if m["role"] == "user"), "")
                response = next((m["content"] for m in messages if m["role"] == "assistant"), "")
                if query and response:
                    train_examples.append(InputExample(texts=[query, response], label=1.0))
    
    logger.info(f"Loaded {len(train_examples)} training examples")
    
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Use MultipleNegativesRankingLoss (works well for retrieval)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Fine-tune
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output = Path(output_dir) / f"{model_name.split('/')[-1]}_{timestamp}"
    
    logger.info(f"Starting fine-tuning for {epochs} epochs")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_ratio=warmup_ratio,
        optimizer_params={"lr": learning_rate},
        output_path=str(model_output),
        show_progress_bar=True,
    )
    
    logger.info(f"Model saved to {model_output}")
    return str(model_output)


def evaluate_embedding_model(
    model_name_or_path: str,
    eval_data_path: str = None,
    tasks: List[str] = None,
):
    """Evaluate embedding model quality."""
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, InformationRetrievalEvaluator
    except ImportError:
        logger.error("sentence-transformers not installed.")
        raise
    
    logger.info(f"Loading model: {model_name_or_path}")
    model = SentenceTransformer(model_name_or_path)
    
    results = {"model": model_name_or_path}
    
    # 1. Semantic similarity evaluation
    if eval_data_path and Path(eval_data_path).exists():
        try:
            from datasets import load_dataset
            # Use STS benchmark
            sts_dataset = load_dataset("glue", "stsb", split="validation")
            
            sentences1 = sts_dataset["sentence1"]
            sentences2 = sts_dataset["sentence2"]
            scores = [float(s) / 5.0 for s in sts_dataset["label"]]  # Normalize to 0-1
            
            evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
            sim_score = evaluator(model)
            results["semantic_similarity"] = sim_score
        except Exception as e:
            logger.warning(f"Semantic similarity evaluation failed: {e}")
    
    # 2. Custom fintech retrieval evaluation
    fintech_queries = {
        "risk_assessment": "How do I assess credit risk for P2P lending?",
        "regulatory_compliance": "What are the SOX compliance requirements for fintech?",
        "algorithmic_trading": "Implement a moving average crossover strategy in Python",
        "fraud_detection": "Design a fraud detection system for online payments",
        "portfolio_optimization": "How to optimize a portfolio using Markowitz theory?",
    }
    
    fintech_documents = [
        "Credit risk assessment involves evaluating borrower creditworthiness through scoring models, analyzing debt-to-income ratios, employment history, and payment behavior.",
        "SOX compliance requires establishing internal controls, audit committees, CEO/CFO certification of financial reports, and regular internal audits for fintech companies.",
        "Moving average crossover strategy generates buy signals when short MA crosses above long MA and sell signals when it crosses below, implemented with pandas.",
        "Fraud detection systems use machine learning models like isolation forests and autoencoders to identify anomalous transaction patterns in real-time.",
        "Markowitz portfolio optimization maximizes the Sharpe ratio by finding the optimal asset allocation using mean-variance analysis and quadratic programming.",
    ]
    
    # Compute embeddings and check retrieval accuracy
    query_embeddings = model.encode(list(fintech_queries.values()))
    doc_embeddings = model.encode(fintech_documents)
    
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_embeddings, doc_embeddings)
    
    # Check if top-1 retrieval is correct (diagonal should be highest)
    correct = sum(1 for i in range(len(fintech_queries)) if similarities[i].argmax() == i)
    retrieval_accuracy = correct / len(fintech_queries)
    
    results["fintech_retrieval_accuracy"] = retrieval_accuracy
    results["fintech_retrieval_details"] = {
        query: {
            "top_doc_idx": int(similarities[i].argmax()),
            "top_similarity": float(similarities[i].max()),
            "correct": int(similarities[i].argmax()) == i,
        }
        for i, query in enumerate(fintech_queries.keys())
    }
    
    logger.info(f"Embedding evaluation results: {results}")
    return results


def compare_embeddings(models: List[str], eval_data_path: str = None):
    """Compare multiple embedding models."""
    all_results = {}
    
    for model_key in models:
        model_info = EMBEDDING_MODELS.get(model_key, {"name": model_key})
        model_name = model_info["name"]
        
        logger.info(f"\nEvaluating: {model_name}")
        results = evaluate_embedding_model(model_name, eval_data_path)
        all_results[model_key] = results
    
    # Print comparison
    print("\n" + "=" * 70)
    print("EMBEDDING MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<40} {'FinTech Retrieval':>18} {'Sim Similarity':>15}")
    print("-" * 70)
    
    for model_key, results in all_results.items():
        retrieval = results.get("fintech_retrieval_accuracy", "N/A")
        similarity = results.get("semantic_similarity", "N/A")
        retrieval_str = f"{retrieval:.2%}" if isinstance(retrieval, float) else retrieval
        similarity_str = f"{similarity:.4f}" if isinstance(similarity, float) else similarity
        print(f"{model_key:<40} {retrieval_str:>18} {similarity_str:>15}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Embedding Model Fine-Tuning & Evaluation")
    parser.add_argument("--model", default=None, help="Model name or key from EMBEDDING_MODELS")
    parser.add_argument("--data", default=None, help="Training/evaluation data path")
    parser.add_argument("--train", action="store_true", help="Fine-tune the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--compare", action="store_true", help="Compare multiple models")
    parser.add_argument("--models", nargs="+", default=["bge-large", "e5-large", "minilm"],
                        help="Models to compare")
    parser.add_argument("--output-dir", default="./embedding_outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable Embedding Models:")
        print("-" * 80)
        for key, info in EMBEDDING_MODELS.items():
            print(f"  {key:<15} {info['name']:<35} dim={info['dimension']}  VRAM={info['vram_gb']}")
            print(f"  {'':<15} {info['description']}")
        return
    
    # Resolve model name
    model_name = args.model
    if model_name and model_name in EMBEDDING_MODELS:
        model_name = EMBEDDING_MODELS[model_name]["name"]
    
    if args.train:
        if not model_name or not args.data:
            parser.error("--model and --data required for training")
        finetune_embedding_model(
            model_name=model_name,
            train_data_path=args.data,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    
    if args.evaluate:
        if not model_name:
            parser.error("--model required for evaluation")
        evaluate_embedding_model(model_name, args.data)
    
    if args.compare:
        compare_embeddings(args.models, args.data)


if __name__ == "__main__":
    main()
