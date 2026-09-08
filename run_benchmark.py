"""Extended benchmark orchestrator.

Runs the full metric taxonomy over MMLU and/or LAMBADA and writes one result
file per model plus a comparison summary.

    python run_benchmark.py --benchmark mmlu --questions 5 --subjects stem
    python run_benchmark.py --benchmark lambada --samples 100 --robustness --context
    python run_benchmark.py --benchmark both --calibration --repeats 3

Passes, and what each costs
--------------------------
  main         1 call per item                 always
  calibration  1 extra call per item           --calibration   (cheap: 1 token)
  consistency  (repeats-1) extra per item      --repeats N
  robustness   1 extra per item per variant    --robustness
  context      1 extra per item per ablation   --context       (LAMBADA)
  seeds        1 full extra run per seed       --seeds a,b,c

The cost multiplier is printed before anything is spent, because it is easy to
ask for a sweep that quietly costs 20x the base run.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence

from benchmark_config import (
    MODEL_ARCH,
    DecodingConfig,
    EvaluationConfig,
    hosted_api_config,
)
from config import MODELS, OPENROUTER_API_KEY, RESULTS_DIR
from metrics import aggregate, consistency as consistency_mod
from metrics import context as context_mod
from metrics import quality as quality_mod
from metrics import robustness as robustness_mod
from telemetry_client import (
    probe_capabilities,
    score_next_token,
    score_options,
    stream_completion,
)

RESULTS_V2_DIR = os.path.join(RESULTS_DIR, "v2")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history.json")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def write_history_entry(benchmark: str, model_metrics: List[Dict],
                        comparison: Dict, args, n_items: int) -> Dict:
    """Append an extended run to results/history.json.

    Shares the file with the classic runs and keeps the fields the History tab
    already charts (name / accuracy / avg_response_time), so old entries keep
    rendering. `extended: true` plus `families` marks which metric blocks
    actually carry data, so the History tab can say what a run covered without
    re-reading every result file.
    """
    import time as _time
    from datetime import datetime

    quality_key = "overall_accuracy" if benchmark == "mmlu" else "last_word_accuracy"
    rank_by_model = {
        r["model"]: r for r in (comparison.get("ranking") or [])
    }

    models = []
    for m in model_metrics:
        tq = m.get("task_quality") or {}
        acc = tq.get(quality_key) or 0.0
        total = tq.get("n_questions") or tq.get("n_passages") or 0
        latency = _dig(m, "api_performance.latency.e2e.mean")
        rank = rank_by_model.get(m["model"], {})
        models.append({
            "model": m["model"],
            "name": m["model"].split("/")[-1],
            "accuracy": round(acc * 100, 1),
            "correct": int(round(acc * total)),
            "total": total,
            "avg_response_time": round(latency, 3) if latency else 0,
            "errors": 0,
            # extended-only fields
            "composite_score": rank.get("composite_score"),
            "rank": rank.get("rank"),
            "components_used": rank.get("components_used", []),
            "cost_total_usd": _dig(m, "economics.total_usd"),
            "cost_per_correct_usd": _dig(m, "economics.cost_per_correct_answer_usd"),
            "ttft_mean": _dig(m, "api_performance.latency.ttft.mean"),
            "provider": _dig(m, "provider_capabilities.provider"),
        })

    first = model_metrics[0] if model_metrics else {}
    families = {
        key: bool(first.get(key) and first[key].get("available") is not False)
        for key in ("task_quality", "probability", "consistency", "context",
                    "robustness", "api_performance", "token_efficiency",
                    "economics", "reliability", "tokenization")
        if key in first
    }
    cfg = first.get("config") or {}

    # Full stage blocks travel with the entry, exactly as the web runner does,
    # so the History tab can render the breakdown for this run without reading
    # results/v2 - where a later run would have overwritten it.
    metrics = {m["model"]: m for m in model_metrics}

    entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": f"{benchmark}-v2",
        "extended": True,
        "benchmark": benchmark,
        "samples": n_items,
        "metrics": metrics,
        "params": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "calibration": bool(args.calibration),
            "repeats": args.repeats,
            "robustness": bool(args.robustness),
            "context": bool(args.context),
            "seeds": args.seeds,
        },
        "families": families,
        "comparable": comparison.get("comparable"),
        "parallelism": (cfg.get("parallelism") or {}),
        "models": models,
    }

    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as fh:
                history = json.load(fh)
        except (ValueError, OSError):
            history = []
    history.insert(0, entry)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)
    return entry


def _dig(data: Dict, dotted: str):
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def mmlu_prompt(subject: str, question: str, choices: Sequence[str],
                template: str = "baseline") -> str:
    """Build an MMLU prompt.

    The "baseline" template delegates to the classic evaluator's prompt rather
    than defining its own. Both passes must ask the model the same question:
    otherwise the extended block reports a different accuracy from the ranking
    table for the same run, and neither number can be trusted. The robustness
    templates are then genuine variations *of that baseline*.
    """
    if template == "baseline":
        from evaluate_slm_mmlu import build_mmlu_prompt
        return build_mmlu_prompt(subject, question, choices)

    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip("ABCD", choices)
    )
    tpl = robustness_mod.PROMPT_TEMPLATES.get(
        template, robustness_mod.PROMPT_TEMPLATES["baseline"]
    )
    return tpl.format(
        subject=subject.replace("_", " "), question=question, options=options
    )


def mmlu_scoring_prompt(subject: str, question: str, choices: Sequence[str]) -> str:
    """Prompt for the single-token calibration pass - no reasoning invited."""
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip("ABCD", choices)
    )
    return (
        f"The following is a multiple choice question about "
        f"{subject.replace('_', ' ')}.\n\n{question}\n{options}\n\n"
        "Respond with exactly one letter (A, B, C or D) and nothing else.\nAnswer:"
    )


def lambada_prompt(context: str, few_shot: int = 0) -> str:
    from evaluate_lambada import build_few_shot
    return (
        "You are given a passage from a novel. Your task is to predict the "
        "very next single word that continues the passage. Respond with ONLY "
        "that one word, no punctuation, no explanation, nothing else.\n\n"
        + build_few_shot(few_shot)
        + "Now predict the next word:\n\n"
        f"Passage: {context}\nAnswer:"
    )


# ---------------------------------------------------------------------------
# Single-item execution
# ---------------------------------------------------------------------------

def run_mmlu_item(model: str, task: Dict, api_key: str, params: Dict,
                  template: str = "baseline", with_calibration: bool = False) -> Dict:
    """One MMLU question: reasoning pass, plus optional calibration pass."""
    from evaluate_slm_mmlu import LETTERS, parse_mmlu_response

    prompt = mmlu_prompt(task["subject"], task["question"], task["choices"], template)
    res = stream_completion(model, prompt, api_key, params, request_logprobs=False)
    letter, reasoning = parse_mmlu_response(res["text"])
    correct_letter = LETTERS[task["answer"]]

    record = {
        "subject": task["subject"],
        "category": task["category"],
        "question": task["question"],
        "choices": task["choices"],
        "correct_letter": correct_letter,
        "predicted_letter": letter,
        "correct": letter == correct_letter,
        "reasoning": reasoning,
        "ttft": res["ttft"],
        "e2e": res["e2e"],
        "time": res["e2e"],
        "prompt_tokens": res["prompt_tokens"],
        "completion_tokens": res["completion_tokens"],
        "reasoning_tokens": res.get("reasoning_tokens"),
        "cached_tokens": res.get("cached_tokens"),
        "cost": res["cost"],
        "provider": res["provider"],
        "error": res["error"],
        "retries": res.get("retries", 0),
        "prompt_template": template,
    }

    if with_calibration:
        sc = score_options(
            model,
            mmlu_scoring_prompt(task["subject"], task["question"], task["choices"]),
            api_key,
        )
        record["scoring_letter"] = sc["predicted_letter"]
        record["scoring_correct"] = sc["predicted_letter"] == correct_letter
        if sc["available"]:
            record["option_probs"] = sc["option_probs"]
            record["correct_option_prob"] = sc["option_probs"][task["answer"]]
        if sc.get("cost"):
            record["cost"] = (record["cost"] or 0) + sc["cost"]
    return record


def run_lambada_item(model: str, passage: Dict, api_key: str, params: Dict,
                     context_override: Optional[str] = None,
                     with_calibration: bool = False) -> Dict:
    """One LAMBADA passage, plus an optional single-token scoring pass.

    The scoring pass is the same trick used for MMLU: at max_tokens=1 the one
    generated token carries a usable top-k, so we recover P(target) and the
    target's rank. With max_tokens>1 the provider returns log-probabilities
    for the final token only, which is the EOS marker and tells us nothing.
    """
    from evaluate_lambada import extract_prediction, normalize_word

    ctx = context_override if context_override is not None else passage["context"]
    prompt = lambada_prompt(ctx, params.get("few_shot", 0))
    res = stream_completion(model, prompt, api_key, params, request_logprobs=False)

    prediction = extract_prediction(res["text"])
    target = normalize_word(passage["target"])

    record = {
        "context": ctx,
        "context_preview": ctx[:200] + ("..." if len(ctx) > 200 else ""),
        "target": passage["target"],
        "prediction": prediction,
        "correct": prediction == target,
        "ttft": res["ttft"],
        "e2e": res["e2e"],
        "time": res["e2e"],
        "prompt_tokens": res["prompt_tokens"],
        "completion_tokens": res["completion_tokens"],
        "reasoning_tokens": res.get("reasoning_tokens"),
        "cached_tokens": res.get("cached_tokens"),
        "cost": res["cost"],
        "provider": res["provider"],
        "error": res["error"],
        "retries": res.get("retries", 0),
    }

    if with_calibration:
        sc = score_next_token(model, prompt, api_key, target)
        if sc.get("cost"):
            record["cost"] = (record["cost"] or 0) + sc["cost"]
        if sc.get("available"):
            record["target_rank"] = sc["target_rank"]
            record["correct_option_prob"] = sc["target_prob"]
            record["option_probs"] = sc["topk_probs"]
    return record


# ---------------------------------------------------------------------------
# Per-model orchestration
# ---------------------------------------------------------------------------

def evaluate_mmlu_model(model: str, tasks: List[Dict], api_key: str,
                        args, run_config) -> Dict:
    params = {
        "temperature": run_config.decoding.temperature,
        "top_p": run_config.decoding.top_p,
        "max_tokens": run_config.decoding.max_tokens,
    }
    start = time.perf_counter()

    print(f"  [{model}] main pass ({len(tasks)} questions)"
          + (" + calibration" if args.calibration else ""))
    records = []
    for i, task in enumerate(tasks, 1):
        records.append(run_mmlu_item(model, task, api_key, params,
                                     with_calibration=args.calibration))
        if i % 10 == 0 or i == len(tasks):
            acc = sum(r["correct"] for r in records) / len(records)
            print(f"    {i}/{len(tasks)} acc={acc:.1%}")

    answer_sets, correct_answers = None, None
    if args.repeats > 1:
        print(f"  [{model}] consistency: {args.repeats - 1} extra repeat(s)")
        answer_sets = [[r["predicted_letter"]] for r in records]
        for _ in range(args.repeats - 1):
            for j, task in enumerate(tasks):
                rec = run_mmlu_item(model, task, api_key, params)
                answer_sets[j].append(rec["predicted_letter"])
        correct_answers = [r["correct_letter"] for r in records]

    robustness_variants = {}
    if args.robustness:
        baseline = [r["correct"] for r in records]
        for variant in ("terse", "verbose"):
            print(f"  [{model}] robustness: prompt_{variant}")
            var = [run_mmlu_item(model, t, api_key, params, template=variant)
                   for t in tasks]
            robustness_variants[f"prompt_{variant}"] = robustness_mod.robustness_delta(
                baseline, [r["correct"] for r in var])

        print(f"  [{model}] robustness: option_reorder")
        reordered = []
        for task in tasks:
            ch, gold = robustness_mod.reorder_options(
                task["choices"], task["answer"], "reverse")
            rec = run_mmlu_item(
                model, {**task, "choices": ch, "answer": gold}, api_key, params)
            reordered.append(rec)
        robustness_variants["option_reorder"] = robustness_mod.robustness_delta(
            baseline, [r["correct"] for r in reordered])

        print(f"  [{model}] robustness: typo_noise")
        noisy = []
        for task in tasks:
            rec = run_mmlu_item(
                model,
                {**task, "question": robustness_mod.typo_perturb(task["question"], 0.08, 7)},
                api_key, params)
            noisy.append(rec)
        robustness_variants["typo_noise"] = robustness_mod.robustness_delta(
            baseline, [r["correct"] for r in noisy])

    runs_by_seed = None
    if args.seeds:
        runs_by_seed = {}
        for seed in args.seeds:
            print(f"  [{model}] seed-stability run seed={seed}")
            p = dict(params, seed=seed)
            runs_by_seed[seed] = [
                run_mmlu_item(model, t, api_key, p)["correct"] for t in tasks
            ]

    wall = time.perf_counter() - start
    return aggregate.build_mmlu_metrics(
        model, records, run_config=run_config, wall_seconds=wall,
        arch=MODEL_ARCH.get(model), answer_sets=answer_sets,
        correct_answers=correct_answers, runs_by_seed=runs_by_seed,
        robustness_variants=robustness_variants,
    )


def evaluate_lambada_model(model: str, passages: List[Dict], api_key: str,
                           args, run_config) -> Dict:
    params = {
        "temperature": run_config.decoding.temperature,
        "top_p": run_config.decoding.top_p,
        "max_tokens": run_config.decoding.max_tokens,
        "few_shot": run_config.decoding.few_shot,
    }
    start = time.perf_counter()

    print(f"  [{model}] main pass ({len(passages)} passages)")
    records = [run_lambada_item(model, p, api_key, params,
                                with_calibration=args.calibration)
               for p in passages]

    ablation_results = None
    if args.context:
        ablation_results = {"full": [r["correct"] for r in records]}
        for name in ("last_sentence", "last_10_words", "no_context"):
            print(f"  [{model}] context ablation: {name}")
            fn = context_mod.CONTEXT_ABLATIONS[name]
            ablation_results[name] = [
                run_lambada_item(model, p, api_key, params,
                                 context_override=fn(p["context"]))["correct"]
                for p in passages
            ]

    robustness_variants = {}
    if args.robustness:
        baseline = [r["correct"] for r in records]
        for name, fn in (("typo", robustness_mod.typo_perturb),
                         ("casing", robustness_mod.casing_noise)):
            print(f"  [{model}] robustness: {name}")
            var = [
                run_lambada_item(model, p, api_key, params,
                                 context_override=fn(p["context"], seed=11))["correct"]
                for p in passages
            ]
            robustness_variants[name] = robustness_mod.robustness_delta(baseline, var)

    answer_sets, correct_answers = None, None
    if args.repeats > 1:
        print(f"  [{model}] consistency: {args.repeats - 1} extra repeat(s)")
        answer_sets = [[r["prediction"]] for r in records]
        for _ in range(args.repeats - 1):
            for j, p in enumerate(passages):
                answer_sets[j].append(
                    run_lambada_item(model, p, api_key, params)["prediction"])
        correct_answers = [
            quality_mod.normalise_word(r["target"]) for r in records
        ]

    wall = time.perf_counter() - start
    return aggregate.build_lambada_metrics(
        model, records, run_config=run_config, wall_seconds=wall,
        arch=MODEL_ARCH.get(model), answer_sets=answer_sets,
        correct_answers=correct_answers,
        robustness_variants=robustness_variants,
        ablation_results=ablation_results,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def estimate_calls(n_items: int, n_models: int, args, benchmark: str) -> int:
    per_model = n_items
    if args.calibration:
        per_model += n_items
    if args.repeats > 1:
        per_model += n_items * (args.repeats - 1)
    if args.robustness:
        per_model += n_items * (4 if benchmark == "mmlu" else 2)
    if args.context and benchmark == "lambada":
        per_model += n_items * 3
    if args.seeds:
        per_model += n_items * len(args.seeds)
    return per_model * n_models


def main():
    ap = argparse.ArgumentParser(description="Extended SLM benchmark")
    ap.add_argument("--benchmark", choices=["mmlu", "lambada", "both"], default="mmlu")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--subjects", default="stem")
    ap.add_argument("--questions", type=int, default=5, help="MMLU questions/subject")
    ap.add_argument("--samples", type=int, default=50, help="LAMBADA passages")
    ap.add_argument("--calibration", action="store_true",
                    help="extra single-token pass for ECE/Brier/NLL")
    ap.add_argument("--repeats", type=int, default=1, help="repeats for stability")
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--context", action="store_true", help="LAMBADA ablations")
    ap.add_argument("--seeds", default="", help="comma-separated seeds")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--few-shot", type=int, default=3)
    ap.add_argument("--yes", action="store_true", help="skip the cost prompt")
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set (see .env.example)")
        return 1

    models = args.models or MODELS
    os.makedirs(RESULTS_V2_DIR, exist_ok=True)

    print("=" * 70)
    print("  SLM Benchmark - extended metric taxonomy")
    print("=" * 70)

    print("\n[1/4] Probing provider capabilities...")
    caps = probe_capabilities(models, OPENROUTER_API_KEY)
    for model, cap in caps.items():
        flags = [k for k in ("logprobs", "streaming_ttft", "usage_accounting",
                             "cost_accounting") if cap.get(k)]
        print(f"  {model:38s} via {str(cap['provider']):12s} -> {', '.join(flags) or 'unreachable'}")
    no_lp = [m for m, c in caps.items() if not c["logprobs"]]
    if no_lp:
        print(f"  NOTE: {len(no_lp)}/{len(models)} model(s) expose no logprobs; "
              "their calibration block will be marked unavailable.")

    benchmarks = ["mmlu", "lambada"] if args.benchmark == "both" else [args.benchmark]
    all_results = {}

    for benchmark in benchmarks:
        print(f"\n[2/4] Loading {benchmark} data...")
        if benchmark == "mmlu":
            from evaluate_slm_mmlu import load_mmlu_tasks, resolve_subjects
            subjects = resolve_subjects(args.subjects)
            items = load_mmlu_tasks(subjects, args.questions)
        else:
            from config import DATASET_FILES
            from evaluate_lambada import load_dataset
            path = DATASET_FILES.get("test")
            if not path or not os.path.exists(path):
                print(f"  LAMBADA dataset missing at {path}; skipping.")
                continue
            items = load_dataset(path, args.samples)
        print(f"  {len(items)} items")

        n_calls = estimate_calls(len(items), len(models), args, benchmark)
        print(f"\n  Estimated API calls: {n_calls} "
              f"({n_calls / max(len(items) * len(models), 1):.1f}x base)")
        if not args.yes:
            if input("  Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("  aborted.")
                return 0

        print(f"\n[3/4] Evaluating {benchmark}...")
        model_metrics = []
        for model in models:
            cfg = hosted_api_config(model, benchmark)
            cfg.decoding = DecodingConfig(
                temperature=args.temperature, max_tokens=args.max_tokens,
                few_shot=args.few_shot,
            )
            cfg.evaluation = EvaluationConfig(
                request_logprobs=args.calibration,
                repeats_for_consistency=args.repeats,
                seeds_for_stability=args.seeds,
                robustness_variants=["prompt", "reorder", "typo"] if args.robustness else [],
                context_ablations=["last_sentence", "last_10_words", "no_context"]
                if args.context else [],
            )
            cfg.dataset.n_items = len(items)
            for problem in cfg.validate():
                print(f"  config warning: {problem}")

            print(f"\n  --- {model} ---")
            if benchmark == "mmlu":
                m = evaluate_mmlu_model(model, items, OPENROUTER_API_KEY, args, cfg)
            else:
                m = evaluate_lambada_model(model, items, OPENROUTER_API_KEY, args, cfg)
            m["provider_capabilities"] = caps.get(model)
            model_metrics.append(m)

            out = os.path.join(
                RESULTS_V2_DIR, f"{model.replace('/', '_')}_{benchmark}_v2.json")
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(m, fh, indent=2, ensure_ascii=False)
            print(f"  saved -> {out}")

        comparison = aggregate.build_comparison(model_metrics, benchmark)
        comparison["provider_capabilities"] = caps
        cmp_path = os.path.join(RESULTS_V2_DIR, f"comparison_{benchmark}.json")
        with open(cmp_path, "w", encoding="utf-8") as fh:
            json.dump(comparison, fh, indent=2, ensure_ascii=False)
        all_results[benchmark] = comparison

        entry = write_history_entry(benchmark, model_metrics, comparison,
                                    args, len(items))
        print(f"  history entry {entry['id']} -> {HISTORY_PATH}")

        print(f"\n[4/4] {benchmark.upper()} ranking")
        print(f"  {'#':<3}{'model':<38}{'quality':>9}{'composite':>11}")
        for row in comparison["ranking"]:
            print(f"  {row['rank']:<3}{row['model']:<38}"
                  f"{row['quality']:>9.3f}{row['composite_score']:>11.3f}")
        if not comparison["comparable"]:
            print(f"  {comparison['comparability_note']}")

    print(f"\nResults in {RESULTS_V2_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
