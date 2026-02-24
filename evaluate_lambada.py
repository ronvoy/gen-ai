"""
LAMBADA Benchmark Evaluation Script

Evaluates language models on the LAMBADA word-prediction task
using the OpenRouter API.
"""

import os
import json
import time
import random
import re
import requests

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODELS,
    TEMPERATURE,
    MAX_TOKENS,
    NUM_SAMPLES,
    DATASET_FILES,
    RESULTS_DIR,
)


def load_dataset(filepath, num_samples=None, seed=42):
    """Load LAMBADA passages and split into context + target word."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if num_samples and num_samples < len(lines):
        random.seed(seed)
        lines = random.sample(lines, num_samples)

    passages = []
    for line in lines:
        words = line.rsplit(" ", 1)
        if len(words) == 2:
            context, target = words
            passages.append({"context": context, "target": target})

    return passages


def normalize_word(word):
    """Lowercase and strip surrounding punctuation for fair comparison."""
    return re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", word.lower())


def query_model(model, context, api_key, temperature=0.0, max_tokens=10):
    """Send a single LAMBADA prompt to OpenRouter and return the prediction."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/lambada-eval",
    }

    prompt = (
        "Complete the following passage with the next single word. "
        "Only respond with that one word, nothing else.\n\n"
        f"Passage: {context}"
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.time()
    try:
        resp = requests.post(
            OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=120
        )
        elapsed = time.time() - start
        resp.raise_for_status()
        data = resp.json()

        raw = data["choices"][0]["message"]["content"].strip()
        # Some models wrap the answer in <think>...</think> tags; strip those
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Take only the first word
        prediction = raw.split()[0] if raw.split() else ""
        prediction = normalize_word(prediction)
        return prediction, elapsed, None
    except Exception as e:
        elapsed = time.time() - start
        return "", elapsed, str(e)


def evaluate_model(model, passages, api_key, temperature=0.0, max_tokens=10):
    """Run LAMBADA evaluation for one model across all sampled passages."""
    results = []
    correct = 0
    total = 0
    total_time = 0.0
    errors = 0

    for i, p in enumerate(passages):
        prediction, elapsed, error = query_model(
            model, p["context"], api_key, temperature, max_tokens
        )
        target = normalize_word(p["target"])
        is_correct = prediction == target

        if is_correct:
            correct += 1
        total += 1
        total_time += elapsed
        if error:
            errors += 1

        results.append(
            {
                "index": i,
                "context_preview": (
                    p["context"][:200] + "..."
                    if len(p["context"]) > 200
                    else p["context"]
                ),
                "target": p["target"],
                "prediction": prediction,
                "correct": is_correct,
                "time": round(elapsed, 3),
                "error": error,
            }
        )

        if (i + 1) % 10 == 0:
            print(
                f"  [{model}] {i + 1}/{len(passages)} "
                f"- Accuracy so far: {correct / total:.2%}"
            )

    accuracy = correct / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0

    return {
        "model": model,
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "avg_response_time": round(avg_time, 3),
        "total_time": round(total_time, 2),
        "errors": errors,
        "results": results,
    }


def run_evaluation(split="test"):
    """Evaluate every configured model on the chosen LAMBADA split."""
    filepath = DATASET_FILES.get(split)
    if not filepath or not os.path.exists(filepath):
        print(f"Dataset file not found for split '{split}': {filepath}")
        return []

    print(f"\n{'=' * 60}")
    print(f"LAMBADA Evaluation - Split: {split}")
    print(f"{'=' * 60}")

    passages = load_dataset(filepath, NUM_SAMPLES)
    print(f"Loaded {len(passages)} passages from {split} split")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    for model in MODELS:
        print(f"\nEvaluating: {model}")
        print("-" * 40)
        result = evaluate_model(model, passages, OPENROUTER_API_KEY, TEMPERATURE, MAX_TOKENS)
        all_results.append(result)

        safe_name = model.replace("/", "_")
        output_path = os.path.join(RESULTS_DIR, f"{safe_name}_lambada_{split}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Results saved to {output_path}")
        print(f"  Accuracy: {result['accuracy']:.2%}")
        print(f"  Avg Response Time: {result['avg_response_time']:.3f}s")

    summary = {
        "split": split,
        "num_samples": NUM_SAMPLES,
        "models": [
            {
                "model": r["model"],
                "accuracy": r["accuracy"],
                "correct": r["correct"],
                "total": r["total"],
                "avg_response_time": r["avg_response_time"],
                "errors": r["errors"],
            }
            for r in all_results
        ],
    }

    summary_path = os.path.join(RESULTS_DIR, f"summary_{split}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return all_results


if __name__ == "__main__":
    run_evaluation("test")
