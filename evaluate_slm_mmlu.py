"""
MMLU benchmark for Small Language Models.

Fetches multiple-choice questions for any of the 57 MMLU subjects from the
free Hugging Face datasets-server API (no API key needed, cached locally),
asks each configured model to answer with explicit step-by-step reasoning,
parses BOTH the final answer letter and the reasoning text, evaluates the
quality/consistency of that reasoning, and ranks the models across several
performance dimensions (accuracy, per-category accuracy, speed, reasoning
consistency, composite score).

Run directly:  python evaluate_slm_mmlu.py
or through the web app (passenger_wsgi.py -> /mmlu).
"""

import os
import json
import re
import time

import requests

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODELS,
    MODEL_INFO,
    RESULTS_DIR,
)

# ===========================================================================
# RUN PARAMETERS - edit these before running the script directly.
# ===========================================================================

# SUBJECT_SELECTION - which subjects to evaluate. Available options:
#   "all"              -> all 57 MMLU subjects
#   "stem"             -> 18 STEM subjects (math, physics, CS, ...)
#   "humanities"       -> 13 humanities subjects (history, law, philosophy, ...)
#   "social_sciences"  -> 12 social science subjects (economics, psychology, ...)
#   "other"            -> 14 other subjects (health, business, misc, ...)
#   or an explicit list of subject names from MMLU_SUBJECTS below, e.g.
#   ["astronomy", "philosophy", "marketing"]
SUBJECT_SELECTION = "stem"

# QUESTIONS_PER_SUBJECT - how many questions to ask per subject.
#   Options: any integer 1..100 (100 is the per-request limit of the free
#   Hugging Face datasets-server API; questions are taken deterministically
#   from the start of the test split so runs are repeatable).
QUESTIONS_PER_SUBJECT = 5

# MODELS_TO_RUN - which models to benchmark. Options:
#   "all"              -> every model in config.MODELS
#   or an explicit list of OpenRouter ids, e.g.
#   ["microsoft/phi-4-mini-instruct"]
MODELS_TO_RUN = "all"

# Decoding parameters used for MMLU (independent from the LAMBADA defaults:
# the model must produce a few sentences of reasoning, not a single word).
#   temperature: 0.0..2.0 (0.0 = deterministic, best for benchmarking)
#   top_p:       0.0..1.0
#   max_tokens:  64..1024 (needs room for the reasoning text)
MMLU_TEMPERATURE = 0.0
MMLU_TOP_P = 1.0
MMLU_MAX_TOKENS = 384

# ===========================================================================
# The 57 MMLU subjects, mapped to their official category
# (categories follow Hendrycks et al., "Measuring Massive Multitask
#  Language Understanding", ICLR 2021).
# ===========================================================================

MMLU_SUBJECTS = {
    # --- STEM (18) ---
    "abstract_algebra": "stem",
    "astronomy": "stem",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_mathematics": "stem",
    "high_school_physics": "stem",
    "high_school_statistics": "stem",
    "machine_learning": "stem",
    # --- Humanities (13) ---
    "formal_logic": "humanities",
    "high_school_european_history": "humanities",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_law": "humanities",
    "world_religions": "humanities",
    # --- Social sciences (12) ---
    "econometrics": "social_sciences",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_microeconomics": "social_sciences",
    "high_school_psychology": "social_sciences",
    "human_sexuality": "social_sciences",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    # --- Other: health, business, misc (14) ---
    "anatomy": "other",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_medicine": "other",
    "global_facts": "other",
    "human_aging": "other",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "nutrition": "other",
    "professional_accounting": "other",
    "professional_medicine": "other",
    "virology": "other",
}

CATEGORY_LABELS = {
    "stem": "STEM",
    "humanities": "Humanities",
    "social_sciences": "Social Sciences",
    "other": "Other",
}

SUBJECT_GROUPS = {
    group: sorted(s for s, c in MMLU_SUBJECTS.items() if c == group)
    for group in CATEGORY_LABELS
}
SUBJECT_GROUPS["all"] = sorted(MMLU_SUBJECTS)

# Free content sources, tried in order. Both serve MMLU through the public
# Hugging Face datasets-server rows API (no key, no login), one config per
# subject, identical row schema: question / choices / answer(index).
MMLU_SOURCES = [
    "cais/mmlu",
    "tasksource/mmlu",
]
DATASETS_SERVER_URL = "https://datasets-server.huggingface.co/rows"
MMLU_CACHE_DIR = os.path.join("_rsc", "mmlu-dataset")

LETTERS = ["A", "B", "C", "D"]

MIN_QUESTIONS = 1
MAX_QUESTIONS = 100


def mmlu_default_params():
    """Decoding parameters for MMLU runs."""
    return {
        "temperature": MMLU_TEMPERATURE,
        "top_p": MMLU_TOP_P,
        "max_tokens": MMLU_MAX_TOKENS,
    }


def resolve_subjects(selection):
    """Turn a selection ("all", a group name, or a list) into subject names."""
    if isinstance(selection, str):
        key = selection.strip().lower()
        if key in SUBJECT_GROUPS:
            return list(SUBJECT_GROUPS[key])
        selection = [key]
    subjects = []
    for s in selection:
        name = str(s).strip().lower()
        if name in MMLU_SUBJECTS and name not in subjects:
            subjects.append(name)
    return subjects


# ---------------------------------------------------------------------------
# Dataset fetching (free Hugging Face datasets-server API, cached locally)
# ---------------------------------------------------------------------------

def _cache_path(subject, base_dir=""):
    directory = os.path.join(base_dir, MMLU_CACHE_DIR) if base_dir else MMLU_CACHE_DIR
    return os.path.join(directory, f"{subject}.json")


def fetch_subject_questions(subject, num_questions, base_dir=""):
    """Return up to `num_questions` test questions for one subject.

    Downloads the first 100 rows of the subject's test split from the free
    datasets-server API on first use and caches them under _rsc/mmlu-dataset,
    so later runs work offline and stay deterministic.
    """
    if subject not in MMLU_SUBJECTS:
        raise ValueError(f"Unknown MMLU subject: {subject}")
    num_questions = max(MIN_QUESTIONS, min(int(num_questions), MAX_QUESTIONS))

    path = _cache_path(subject, base_dir)
    rows = None
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f).get("rows")
        except (ValueError, OSError):
            rows = None

    if not rows:
        rows, source = _download_subject(subject)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"subject": subject, "source": source, "rows": rows}, f,
                      ensure_ascii=False)

    return rows[:num_questions]


def _download_subject(subject):
    """Fetch 100 rows for a subject, trying each free source in order."""
    last_error = None
    for dataset in MMLU_SOURCES:
        try:
            resp = requests.get(
                DATASETS_SERVER_URL,
                params={
                    "dataset": dataset,
                    "config": subject,
                    "split": "test",
                    "offset": 0,
                    "length": MAX_QUESTIONS,
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw_rows = resp.json().get("rows", [])
            rows = []
            for item in raw_rows:
                row = item.get("row", {})
                question = row.get("question")
                choices = row.get("choices")
                answer = row.get("answer")
                if question and isinstance(choices, list) and len(choices) == 4:
                    rows.append({
                        "question": question.strip(),
                        "choices": [str(c).strip() for c in choices],
                        "answer": int(answer),
                    })
            if rows:
                return rows, dataset
            last_error = f"{dataset}: empty response"
        except Exception as e:
            last_error = f"{dataset}: {e}"
    raise RuntimeError(
        f"Could not fetch MMLU subject '{subject}' from any free source "
        f"({last_error})"
    )


def load_mmlu_tasks(subjects, questions_per_subject, base_dir=""):
    """Load questions for every subject into a flat, ordered task list."""
    tasks = []
    for subject in subjects:
        for i, row in enumerate(
            fetch_subject_questions(subject, questions_per_subject, base_dir)
        ):
            tasks.append({
                "subject": subject,
                "category": MMLU_SUBJECTS[subject],
                "index": i,
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],  # 0..3 index of the correct choice
            })
    return tasks


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# A single worked example teaches small models the exact output format, which
# makes both the answer letter and the reasoning reliably parseable. Asking
# for the reasoning BEFORE the letter is deliberate chain-of-thought: the
# letter is then conditioned on the reasoning instead of being a blind guess.
WORKED_EXAMPLE = (
    "Question: Which planet is known as the Red Planet?\n"
    "A. Venus\n"
    "B. Mars\n"
    "C. Jupiter\n"
    "D. Mercury\n"
    "Reasoning: The Red Planet nickname comes from the reddish iron-oxide "
    "dust covering the surface. That describes Mars, not Venus, Jupiter or "
    "Mercury.\n"
    "Answer: B\n"
)


def build_mmlu_prompt(subject, question, choices):
    """Build the chain-of-thought prompt for a single MMLU question."""
    topic = subject.replace("_", " ")
    options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"You are an expert in {topic} answering one multiple-choice question.\n"
        "First think through the problem step by step in 1-4 short sentences, "
        "then commit to exactly one option. Use exactly this output format and "
        "nothing else:\n"
        "Reasoning: <your step-by-step reasoning>\n"
        "Answer: <one letter: A, B, C, or D>\n\n"
        "Here is a worked example of the format:\n"
        f"{WORKED_EXAMPLE}\n"
        "Now answer this question:\n"
        f"Question: {question}\n"
        f"{options}\n"
    )


# ---------------------------------------------------------------------------
# Response parsing: final letter + reasoning text
# ---------------------------------------------------------------------------

def parse_mmlu_response(raw_text):
    """Split a model response into (answer_letter, reasoning_text).

    Handles: the requested "Reasoning:/Answer:" format, <think>...</think>
    blocks from reasoning models, "The answer is (B)" phrasing, a bare
    letter, and truncated responses. Returns ("", reasoning) when no letter
    can be found.
    """
    text = (raw_text or "").strip()
    if not text:
        return "", ""

    # Reasoning models emit their thinking in <think> tags: keep it as
    # reasoning and parse the visible part for the letter.
    thinking = " ".join(
        m.group(1).strip()
        for m in re.finditer(r"<think>(.*?)(?:</think>|$)", text, re.DOTALL)
    ).strip()
    visible = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()
    search_space = visible or thinking

    # 1) Explicit "Answer: X" (possibly bold/parenthesised) - last occurrence.
    letter = ""
    matches = re.findall(
        r"(?:final\s+answer|answer)\s*(?:is)?\s*[:\-]?\s*\**\s*\(?([A-Da-d])\)?\b",
        search_space,
    )
    if matches:
        letter = matches[-1].upper()

    # 2) Response that starts with a bare letter, e.g. "B." or "(C) because...".
    if not letter:
        m = re.match(r"^\s*\(?([A-Da-d])\)?\s*[\).:,\-]?(\s|$)", search_space)
        if m:
            letter = m.group(1).upper()

    # 3) Last standalone capital letter A-D anywhere in the response.
    if not letter:
        standalone = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", search_space)
        if standalone:
            letter = standalone[-1]

    # Reasoning: prefer the explicit "Reasoning:" section, else everything
    # before/around the answer line; prepend <think> content when present.
    m = re.search(
        r"reasoning\s*[:\-]\s*(.*?)(?=\n\s*\**\s*(?:final\s+answer|answer)\s*[:\-]|\Z)",
        visible, re.IGNORECASE | re.DOTALL,
    )
    if m:
        reasoning = m.group(1).strip()
    else:
        reasoning = re.sub(
            r"\**\s*(?:final\s+answer|answer)\s*(?:is)?\s*[:\-]?\s*\**\s*\(?[A-Da-d]\)?\.?\s*$",
            "", visible, flags=re.IGNORECASE,
        ).strip()
    if thinking:
        reasoning = (thinking + ("\n" + reasoning if reasoning else "")).strip()

    return letter, reasoning


def analyze_reasoning(reasoning, letter, choices, correct_letter):
    """Heuristic quality check of the model's reasoning for one question.

    - has_reasoning: the model produced a non-trivial explanation
    - consistent: the reasoning actually supports the chosen option (it names
      the option letter or reuses the meaningful words of the chosen choice)
    - verdict: a small label used for the per-question display and the
      aggregate reasoning metrics
    """
    words = len(reasoning.split()) if reasoning else 0
    has_reasoning = words >= 5

    consistent = False
    if has_reasoning and letter in LETTERS:
        low = reasoning.lower()
        if re.search(rf"\b(option|choice|answer|it)\s*(?:is)?\s*\(?{letter.lower()}\)?\b", low):
            consistent = True
        else:
            choice_text = choices[LETTERS.index(letter)].lower()
            keywords = [w for w in re.findall(r"[a-z0-9]+", choice_text) if len(w) > 3]
            if keywords:
                overlap = sum(1 for w in keywords if w in low) / len(keywords)
                consistent = overlap >= 0.5

    is_correct = letter == correct_letter
    if not letter:
        verdict = "no-answer"
    elif is_correct and has_reasoning and consistent:
        verdict = "sound"
    elif is_correct and has_reasoning:
        verdict = "right-weak-link"   # correct, but reasoning doesn't clearly support it
    elif is_correct:
        verdict = "lucky-guess"       # correct with no real reasoning
    elif has_reasoning:
        verdict = "flawed-reasoning"  # reasoned its way to a wrong answer
    else:
        verdict = "blind-guess"       # wrong and no reasoning

    return {
        "has_reasoning": has_reasoning,
        "word_count": words,
        "consistent": consistent,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Model querying and evaluation
# ---------------------------------------------------------------------------

def query_model_mmlu(model, prompt, api_key, params=None):
    """Send one MMLU prompt to OpenRouter; return (raw_text, seconds, error)."""
    if params is None:
        params = mmlu_default_params()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/slm-mmlu-eval",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "max_tokens": params["max_tokens"],
    }

    start = time.time()
    try:
        resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload,
                             timeout=180)
        elapsed = time.time() - start
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return raw, elapsed, None
    except Exception as e:
        return "", time.time() - start, str(e)


def evaluate_model_mmlu(model, tasks, api_key, params=None, progress=None):
    """Evaluate one model on a list of MMLU tasks.

    Returns a result dict with overall, per-subject and per-category accuracy,
    aggregate reasoning metrics, and the full per-question record (question,
    options, model answer, correct answer, reasoning text and its analysis)
    for display in the web app.
    """
    if params is None:
        params = mmlu_default_params()

    per_question = []
    subject_stats = {}
    correct = 0
    errors = 0
    total_time = 0.0
    reasoned = 0
    consistent = 0
    reasoning_words = 0

    for i, task in enumerate(tasks):
        prompt = build_mmlu_prompt(task["subject"], task["question"], task["choices"])
        raw, elapsed, error = query_model_mmlu(model, prompt, api_key, params)
        letter, reasoning = parse_mmlu_response(raw)
        correct_letter = LETTERS[task["answer"]]
        analysis = analyze_reasoning(reasoning, letter, task["choices"], correct_letter)
        is_correct = letter == correct_letter

        correct += int(is_correct)
        errors += int(bool(error))
        total_time += elapsed
        reasoned += int(analysis["has_reasoning"])
        consistent += int(analysis["consistent"])
        reasoning_words += analysis["word_count"]

        stats = subject_stats.setdefault(
            task["subject"],
            {"category": task["category"], "correct": 0, "total": 0},
        )
        stats["total"] += 1
        stats["correct"] += int(is_correct)

        per_question.append({
            "subject": task["subject"],
            "category": task["category"],
            "index": task["index"],
            "question": task["question"],
            "choices": task["choices"],
            "correct_letter": correct_letter,
            "predicted_letter": letter,
            "correct": is_correct,
            "reasoning": reasoning,
            "analysis": analysis,
            "time": round(elapsed, 3),
            "error": error,
        })

        if progress and (i + 1) % 5 == 0:
            progress(model, i + 1, len(tasks), correct)

    total = len(per_question)
    for stats in subject_stats.values():
        stats["accuracy"] = round(stats["correct"] / stats["total"], 4)

    category_stats = {}
    for stats in subject_stats.values():
        cat = category_stats.setdefault(
            stats["category"], {"correct": 0, "total": 0}
        )
        cat["correct"] += stats["correct"]
        cat["total"] += stats["total"]
    for cat in category_stats.values():
        cat["accuracy"] = round(cat["correct"] / cat["total"], 4)

    return {
        "model": model,
        "benchmark": "mmlu",
        "subjects": sorted(subject_stats),
        "questions_per_subject": max(
            (s["total"] for s in subject_stats.values()), default=0
        ),
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0,
        "avg_response_time": round(total_time / total, 3) if total else 0,
        "total_time": round(total_time, 2),
        "errors": errors,
        "subject_accuracy": subject_stats,
        "category_accuracy": category_stats,
        "reasoning": {
            "reasoning_rate": round(reasoned / total, 4) if total else 0,
            "consistency_rate": round(consistent / total, 4) if total else 0,
            "avg_reasoning_words": round(reasoning_words / total, 1) if total else 0,
        },
        "params": params,
        "results": per_question,
    }


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def build_mmlu_summary(results):
    """Rank models across performance dimensions and build the summary dict.

    Per-dimension ranks (1 = best) for accuracy, speed and reasoning
    consistency, plus a composite score:
        composite = 0.70 * accuracy
                  + 0.15 * reasoning consistency rate
                  + 0.15 * relative speed (fastest model / this model)
    Accuracy dominates; reasoning quality and latency break ties, mirroring
    how SLMs are chosen in practice (quality first, then cost/latency).
    """
    results = [r for r in results if r.get("total")]
    if not results:
        return {"benchmark": "mmlu", "ranking": [], "models": []}

    min_time = min(r["avg_response_time"] for r in results) or 1e-9

    def _rank(rows, key, reverse):
        order = sorted(rows, key=key, reverse=reverse)
        return {r["model"]: i + 1 for i, r in enumerate(order)}

    acc_rank = _rank(results, lambda r: r["accuracy"], True)
    speed_rank = _rank(results, lambda r: r["avg_response_time"], False)
    reason_rank = _rank(results, lambda r: r["reasoning"]["consistency_rate"], True)

    ranking = []
    for r in results:
        speed_score = min_time / max(r["avg_response_time"], 1e-9)
        composite = (
            0.70 * r["accuracy"]
            + 0.15 * r["reasoning"]["consistency_rate"]
            + 0.15 * speed_score
        )
        info = MODEL_INFO.get(r["model"], {})
        ranking.append({
            "model": r["model"],
            "name": info.get("name", r["model"].split("/")[-1]),
            "params": info.get("params", ""),
            "accuracy": r["accuracy"],
            "avg_response_time": r["avg_response_time"],
            "consistency_rate": r["reasoning"]["consistency_rate"],
            "reasoning_rate": r["reasoning"]["reasoning_rate"],
            "avg_reasoning_words": r["reasoning"]["avg_reasoning_words"],
            "category_accuracy": {
                c: v["accuracy"] for c, v in r["category_accuracy"].items()
            },
            "composite_score": round(composite, 4),
            "ranks": {
                "accuracy": acc_rank[r["model"]],
                "speed": speed_rank[r["model"]],
                "reasoning": reason_rank[r["model"]],
            },
            "subjects": r["subjects"],
            "total": r["total"],
            "correct": r["correct"],
            "errors": r["errors"],
        })

    ranking.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, row in enumerate(ranking):
        row["rank"] = i + 1

    return {
        "benchmark": "mmlu",
        "ranking": ranking,
        "models": [
            {
                "model": r["model"],
                "accuracy": r["accuracy"],
                "correct": r["correct"],
                "total": r["total"],
                "avg_response_time": r["avg_response_time"],
                "errors": r["errors"],
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _progress(model, done, total, correct):
    print(f"  [{model}] {done}/{total} - accuracy so far: {correct / done:.2%}")


def run_mmlu_evaluation(subject_selection=None, questions_per_subject=None,
                        models=None):
    """Full pipeline: fetch questions, evaluate models, rank, save results."""
    subjects = resolve_subjects(subject_selection or SUBJECT_SELECTION)
    if not subjects:
        print("No valid subjects selected."); return []
    n = questions_per_subject or QUESTIONS_PER_SUBJECT

    if models is None:
        models = MODELS_TO_RUN
    if models == "all" or not isinstance(models, list):
        models = list(MODELS)
    else:
        models = [m for m in models if m in MODELS] or list(MODELS)

    print(f"\n{'=' * 60}\nMMLU Evaluation - {len(subjects)} subject(s), "
          f"{n} question(s) each\n{'=' * 60}")

    tasks = load_mmlu_tasks(subjects, n)
    print(f"Loaded {len(tasks)} questions "
          f"(cached under {MMLU_CACHE_DIR}, source: free HF datasets-server)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []
    for model in models:
        print(f"\nEvaluating: {model}\n{'-' * 40}")
        result = evaluate_model_mmlu(
            model, tasks, OPENROUTER_API_KEY, mmlu_default_params(), _progress
        )
        all_results.append(result)

        safe = model.replace("/", "_")
        path = os.path.join(RESULTS_DIR, f"{safe}_mmlu.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Accuracy: {result['accuracy']:.2%} | "
              f"Reasoning consistency: {result['reasoning']['consistency_rate']:.2%} | "
              f"Avg time: {result['avg_response_time']:.2f}s")
        print(f"  Saved to {path}")

    summary = build_mmlu_summary(all_results)
    summary["subjects"] = subjects
    summary["questions_per_subject"] = n
    with open(os.path.join(RESULTS_DIR, "summary_mmlu.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}\nRanking (composite = 0.70 accuracy + 0.15 reasoning "
          f"consistency + 0.15 speed)\n{'=' * 60}")
    for row in summary["ranking"]:
        print(f"  #{row['rank']} {row['name']:<15} "
              f"acc {row['accuracy']:.2%} (rank {row['ranks']['accuracy']}) | "
              f"reasoning {row['consistency_rate']:.2%} (rank {row['ranks']['reasoning']}) | "
              f"speed {row['avg_response_time']:.2f}s (rank {row['ranks']['speed']}) | "
              f"composite {row['composite_score']:.3f}")
    print(f"\nSummary saved to {os.path.join(RESULTS_DIR, 'summary_mmlu.json')}")
    return all_results


def _parse_cli():
    """CLI flags (used by run_mmlu.sh); defaults fall back to the RUN
    PARAMETERS at the top of this file."""
    import argparse

    parser = argparse.ArgumentParser(description="MMLU benchmark for SLMs")
    parser.add_argument(
        "--subjects", default=None,
        help='"all", a group (stem, humanities, social_sciences, other), '
             'or comma-separated subject names',
    )
    parser.add_argument(
        "--questions", type=int, default=None,
        help="questions per subject (1-100)",
    )
    parser.add_argument(
        "--models", default=None,
        help='"all" or comma-separated OpenRouter model ids',
    )
    args = parser.parse_args()

    subjects = args.subjects
    if subjects and "," in subjects:
        subjects = [s.strip() for s in subjects.split(",") if s.strip()]
    models = args.models
    if models and models != "all":
        models = [m.strip() for m in models.split(",") if m.strip()]
    return subjects, args.questions, models


if __name__ == "__main__":
    cli_subjects, cli_questions, cli_models = _parse_cli()
    run_mmlu_evaluation(cli_subjects, cli_questions, cli_models)
