#!/usr/bin/env python3
"""
LAMBADA Benchmark Evaluation Script
Evaluates three Small Language Models on the LAMBADA dataset via OpenRouter API.

Models evaluated:
  1. Phi-3 Mini (3.8B)   - Microsoft
  2. Gemma 2 9B          - Google DeepMind
  3. Llama 3.1 8B        - Meta

Usage:
  python evaluate_lambada.py --api-key YOUR_KEY [--samples 50] [--benchmark lambada_test]
"""

import os
import json
import time
import random
import argparse
import requests
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "phi-3-mini": {
        "id": "microsoft/phi-3-mini-128k-instruct",
        "name": "Phi-3 Mini",
        "provider": "Microsoft",
        "params": "3.8B",
        "context_window": "128K",
        "architecture": "Dense Transformer Decoder",
    },
    "gemma-2-9b": {
        "id": "google/gemma-2-9b-it:free",
        "name": "Gemma 2 9B",
        "provider": "Google DeepMind",
        "params": "9B",
        "context_window": "8K",
        "architecture": "Transformer Decoder with Local+Global Attention",
    },
    "llama-3.1-8b": {
        "id": "meta-llama/llama-3.1-8b-instruct:free",
        "name": "Llama 3.1 8B",
        "provider": "Meta",
        "params": "8B",
        "context_window": "128K",
        "architecture": "Dense Transformer Decoder with GQA",
    },
}

# ---------------------------------------------------------------------------
# Dataset paths relative to project root
# ---------------------------------------------------------------------------
BENCHMARK_FILES = {
    "lambada_test": "_rsc/lambada-dataset/lambada_test_plain_text.txt",
    "lambada_dev": "_rsc/lambada-dataset/lambada_development_plain_text.txt",
    "lambada_control": "_rsc/lambada-dataset/lambada_control_test_data_plain_text.txt",
    "rejected": "_rsc/rejected-data1/rejected/rejected_plain_text.txt",
}

SYSTEM_PROMPT = (
    "You are a language model being evaluated on next-word prediction. "
    "Given a passage with its last word removed, predict the single missing word. "
    "Reply with ONLY that one word — no punctuation, no explanation."
)

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_passages(file_path: str, sample_size: int = 50, seed: int = 42) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        passages = [line.strip() for line in f if line.strip()]
    random.seed(seed)
    if len(passages) > sample_size:
        passages = random.sample(passages, sample_size)
    return passages


def extract_target(passage: str) -> tuple[str, str]:
    words = passage.split()
    target = words[-1].strip(".,!?;:'\"()[]{}").lower()
    context = " ".join(words[:-1])
    return context, target


def query_openrouter(api_key: str, model_id: str, context: str):
    start = time.time()
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Complete this passage with exactly one word. "
                            "Output ONLY the word:\n\n" + context
                        ),
                    },
                ],
                "max_tokens": 10,
                "temperature": 0,
            },
            timeout=60,
        )
        latency = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()
            prediction = raw.split()[0] if raw else ""
            prediction = prediction.strip(".,!?;:'\"()[]{}").lower()
            return prediction, latency, None
        return "", latency, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:
        return "", time.time() - start, str(exc)


def evaluate_model(
    api_key: str,
    model_key: str,
    passages: list[str],
    benchmark_name: str = "lambada_test",
    progress_callback=None,
):
    info = MODELS[model_key]
    results = []
    correct = 0
    total_latency = 0.0
    errors = 0

    for i, passage in enumerate(passages):
        context, target = extract_target(passage)
        prediction, latency, error = query_openrouter(api_key, info["id"], context)

        is_correct = prediction == target
        if is_correct:
            correct += 1
        if error:
            errors += 1

        total_latency += latency

        results.append({
            "idx": i,
            "context_snippet": context[:300],
            "target": target,
            "prediction": prediction,
            "correct": is_correct,
            "latency_s": round(latency, 3),
            "error": error,
        })

        if progress_callback:
            progress_callback(i + 1, len(passages))

        time.sleep(0.3)

    n = len(passages)
    metrics = {
        "model_key": model_key,
        "model_name": info["name"],
        "model_id": info["id"],
        "provider": info["provider"],
        "params": info["params"],
        "benchmark": benchmark_name,
        "total_passages": n,
        "correct": correct,
        "errors": errors,
        "accuracy": round(correct / n, 4) if n else 0,
        "error_rate": round(errors / n, 4) if n else 0,
        "avg_latency_s": round(total_latency / n, 3) if n else 0,
        "total_time_s": round(total_latency, 2),
        "timestamp": datetime.now().isoformat(),
    }

    return metrics, results


def save_results(metrics: dict, predictions: list, out_dir: str = "results"):
    Path(out_dir).mkdir(exist_ok=True)
    key = metrics["model_key"]
    bench = metrics["benchmark"]
    fname = f"{key}_{bench}.json"
    payload = {"metrics": metrics, "predictions": predictions}
    with open(Path(out_dir) / fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  → Saved {fname}")
    return str(Path(out_dir) / fname)

# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LAMBADA SLM Evaluation")
    parser.add_argument("--api-key", required=True, help="OpenRouter API key")
    parser.add_argument("--samples", type=int, default=50, help="Passages to sample")
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARK_FILES.keys()),
        default="lambada_test",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
    )
    args = parser.parse_args()

    data_path = BENCHMARK_FILES[args.benchmark]
    if not Path(data_path).exists():
        print(f"ERROR: Dataset file not found: {data_path}")
        return

    passages = load_passages(data_path, args.samples)
    print(f"Loaded {len(passages)} passages from {args.benchmark}\n")

    all_metrics = []
    for mk in args.models:
        info = MODELS[mk]
        print(f"Evaluating {info['name']} ({info['params']}) …")

        def _progress(done, total):
            pct = done / total * 100
            print(f"  [{done}/{total}] {pct:.0f}%", end="\r")

        metrics, preds = evaluate_model(
            args.api_key, mk, passages, args.benchmark, _progress
        )
        print(f"\n  Accuracy: {metrics['accuracy']:.2%}  "
              f"| Avg latency: {metrics['avg_latency_s']:.2f}s\n")
        save_results(metrics, preds)
        all_metrics.append(metrics)

    summary_path = Path("results") / f"summary_{args.benchmark}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
