"""Assemble per-model metric blocks into a comparable report structure.

Keeps the taxonomy the project is organised around:

    <benchmark>
      task_quality        - was it right?
      probability         - how sure was it, and was that confidence earned?
      consistency         - would it say the same thing again?
      robustness          - does the score survive a harmless input change?
      context / tokenization  (LAMBADA only)
      systems             - what did it cost in time, memory and money?

`config` sits alongside these, never inside them: configuration is an input,
metrics are outputs, and the composite score is computed from quality and
systems only - never from a parallelism degree.
"""

from typing import Dict, List, Optional, Sequence

from . import calibration, consistency, context, quality, robustness
from . import reliability as reliability_mod
from . import systems as systems_mod
from . import taxonomy
from . import tokenization


# ---------------------------------------------------------------------------
# Per-model assembly
# ---------------------------------------------------------------------------

def build_mmlu_metrics(
    model: str,
    records: Sequence[Dict],
    run_config=None,
    wall_seconds: Optional[float] = None,
    arch: Optional[Dict] = None,
    answer_sets: Optional[Sequence[Sequence[str]]] = None,
    correct_answers: Optional[Sequence[str]] = None,
    runs_by_seed: Optional[Dict[int, Sequence[bool]]] = None,
    robustness_variants: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """Full metric set for one model on MMLU."""
    task_quality = quality.build_mmlu_quality_block(records)

    correct_probs, distributions, correctness, correct_idx, ranks = (
        _extract_probability_inputs(records, n_options=4)
    )
    prob_block = calibration.summarise_probability_block(
        correct_probs, distributions, correctness,
        correct_indices=correct_idx, ranks=ranks,
        n_bins=_bins(run_config),
    )

    n_correct = int(round((task_quality.get("overall_accuracy") or 0)
                          * task_quality.get("n_questions", 0)))

    return {
        "model": model,
        "benchmark": "mmlu",
        # Stages 1-9. Stage 10 (hardware/distributed) is excluded by design -
        # see metrics/taxonomy.EXCLUDED_STAGE.
        "task_quality": task_quality,                                   # 1
        "probability": prob_block,                                      # 2
        "consistency": consistency.build_consistency_block(             # 3
            answer_sets, correct_answers, runs_by_seed),
        "robustness": robustness.build_robustness_block(                # 5
            robustness_variants or {}),
        "api_performance": systems_mod.build_api_performance_block(     # 6
            records, wall_seconds=wall_seconds),
        "token_efficiency": systems_mod.build_token_efficiency_block(   # 7
            records, correct_count=n_correct),
        "economics": systems_mod.build_economics_block(                 # 8
            records, correct_count=n_correct),
        "reliability": reliability_mod.build_reliability_block(         # 9
            records, answer_key="predicted_letter"),
        "config": run_config.to_dict() if hasattr(run_config, "to_dict") else run_config,
        "taxonomy": taxonomy.observability_report(),
    }


def build_lambada_metrics(
    model: str,
    records: Sequence[Dict],
    run_config=None,
    wall_seconds: Optional[float] = None,
    arch: Optional[Dict] = None,
    answer_sets: Optional[Sequence[Sequence[str]]] = None,
    correct_answers: Optional[Sequence[str]] = None,
    runs_by_seed: Optional[Dict[int, Sequence[bool]]] = None,
    robustness_variants: Optional[Dict[str, Dict]] = None,
    ablation_results: Optional[Dict[str, Sequence[bool]]] = None,
) -> Dict:
    """Full metric set for one model on LAMBADA."""
    task_quality = quality.build_lambada_quality_block(records)

    correct_probs, distributions, correctness, _, ranks = (
        _extract_probability_inputs(records, n_options=None)
    )
    prob_block = calibration.summarise_probability_block(
        correct_probs, distributions, correctness,
        ranks=ranks, n_bins=_bins(run_config),
    )

    n_correct = int(round((task_quality.get("last_word_accuracy") or 0)
                          * task_quality.get("n_passages", 0)))

    return {
        "model": model,
        "benchmark": "lambada",
        "task_quality": task_quality,                                   # 1
        "probability": prob_block,                                      # 2
        "consistency": consistency.build_consistency_block(             # 3
            answer_sets, correct_answers, runs_by_seed),
        "context": context.build_context_block(ablation_results, records),  # 4
        "robustness": robustness.build_robustness_block(                # 5
            robustness_variants or {}),
        "api_performance": systems_mod.build_api_performance_block(     # 6
            records, wall_seconds=wall_seconds),
        "token_efficiency": systems_mod.build_token_efficiency_block(   # 7
            records, correct_count=n_correct),
        "economics": systems_mod.build_economics_block(                 # 8
            records, correct_count=n_correct),
        "reliability": reliability_mod.build_reliability_block(         # 9
            records, answer_key="prediction"),
        "tokenization": tokenization.build_tokenization_block(records, model),
        "config": run_config.to_dict() if hasattr(run_config, "to_dict") else run_config,
        "taxonomy": taxonomy.observability_report(),
    }


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

# Composite weights. Quality dominates because a fast wrong answer is worthless;
# calibration is included because an SLM that knows when it is unsure can be
# routed to a bigger model, which is the main way small models are used well.
COMPOSITE_WEIGHTS = {
    "quality": 0.50,
    "calibration": 0.15,
    "robustness": 0.15,
    "efficiency": 0.10,
    # Stage 9. A model whose calls fail or return garbage is unusable no matter
    # how well it scores on the ones that land, so reliability carries weight
    # rather than sitting in a footnote.
    "reliability": 0.10,
}

# Stages that are measured and reported but deliberately kept out of the
# composite. Recorded here (and echoed into the comparison output) so the
# omission reads as a decision rather than an oversight.
COMPONENTS_EXCLUDED_BY_DESIGN = {
    "context": (
        "Stage 4 is LAMBADA-only and its headline figure, context utilisation, "
        "needs the ablation pass. Scoring it would make the MMLU and LAMBADA "
        "composites structurally different weightings under one name."
    ),
    "tokenization": (
        "A tokenizer handicap is a property of the vocabulary, not of the "
        "model's competence, and two of the three tokenizers are gated so the "
        "figures are approximate. It explains a score; it should not be one."
    ),
}


def build_comparison(model_metrics: Sequence[Dict], benchmark: str = "mmlu") -> Dict:
    """Rank models and expose the sub-scores behind the ranking.

    Refuses to rank across incompatible configs: if two models were run with
    different decoding or dataset settings the comparison is confounded, and a
    silent ranking would be worse than none.
    """
    rows = [m for m in model_metrics if m.get("task_quality")]
    if not rows:
        return {"benchmark": benchmark, "ranking": [], "comparable": False}

    config_keys = {
        _comparable_key(m.get("config")) for m in rows if m.get("config")
    }
    comparable = len(config_keys) <= 1

    quality_key = (
        "overall_accuracy" if benchmark == "mmlu" else "last_word_accuracy"
    )
    latencies = [
        _dig(m, "api_performance.latency.e2e.mean") for m in rows
    ]
    latencies = [v for v in latencies if isinstance(v, (int, float)) and v > 0]
    best_latency = min(latencies) if latencies else None

    ranking = []
    for m in rows:
        acc = m["task_quality"].get(quality_key) or 0.0

        # Calibration sub-score: 1 - ECE, only when logprobs were available.
        # Clamped like the other sub-scores: ECE can exceed 1 in principle, and
        # an unclamped negative term would drag the composite below zero and
        # break the 0-1 scale every other component is read on.
        ece = _dig(m, "probability.ece")
        cal_score = (max(0.0, min(1.0, 1.0 - ece))
                     if isinstance(ece, (int, float)) else None)

        rob_score = _dig(m, "robustness.robustness_score")

        # Efficiency is RELATIVE to the fastest model in this run, so 1.0 means
        # "quickest here", not "quick". Adding or removing a model rebases it,
        # which is why efficiency (and any composite containing it) must not be
        # compared across runs.
        lat = _dig(m, "api_performance.latency.e2e.mean")
        eff_score = (
            best_latency / lat
            if best_latency and isinstance(lat, (int, float)) and lat > 0
            else None
        )

        # Reliability: successful calls that also returned something usable.
        success = _dig(m, "reliability.success_rate")
        invalid = _dig(m, "reliability.invalid_output_rate")
        rel_score = None
        if isinstance(success, (int, float)):
            rel_score = success - (invalid if isinstance(invalid, (int, float)) else 0.0)
            rel_score = max(0.0, min(1.0, rel_score))

        parts = {
            "quality": acc,
            "calibration": cal_score,
            "robustness": rob_score,
            "efficiency": eff_score,
            "reliability": rel_score,
        }
        # Renormalise over whichever components are actually available, so a
        # model is never penalised for a study we did not run.
        available = {k: v for k, v in parts.items() if v is not None}
        weight_sum = sum(COMPOSITE_WEIGHTS[k] for k in available) or 1.0
        composite = sum(
            COMPOSITE_WEIGHTS[k] * v for k, v in available.items()
        ) / weight_sum

        ranking.append({
            "model": m["model"],
            "quality": round(acc, 4),
            "calibration_score": _r(cal_score),
            "robustness_score": _r(rob_score),
            "efficiency_score": _r(eff_score),
            "reliability_score": _r(rel_score),
            "components_used": [k for k in COMPOSITE_WEIGHTS if k in available],
            # The weights that were *actually applied* after renormalisation.
            # Without these a reader recomputing the composite from
            # COMPOSITE_WEIGHTS gets a different number and cannot tell why.
            # Keyed in COMPOSITE_WEIGHTS order, not alphabetically, so the
            # printed formula reads quality-first like the nominal one.
            "effective_weights": {
                k: round(COMPOSITE_WEIGHTS[k] / weight_sum, 4)
                for k in COMPOSITE_WEIGHTS if k in available
            },
            "composite_score": round(composite, 4),
            "mean_latency_s": _r(lat),
            "ttft_mean_s": _r(_dig(m, "api_performance.latency.ttft.mean")),
            "cost_total_usd": _dig(m, "economics.total_usd"),
            "cost_per_correct_usd": _dig(m, "economics.cost_per_correct_answer_usd"),
            "cost_per_1m_tokens_usd": _dig(m, "economics.cost_per_1m_tokens_usd"),
            "total_tokens": _dig(m, "token_efficiency.total_tokens"),
            "success_rate": _dig(m, "reliability.success_rate"),
        })

    ranking.sort(key=lambda r: r["composite_score"], reverse=True)
    for i, row in enumerate(ranking, start=1):
        row["rank"] = i

    # Renormalisation is only fair when every model renormalises over the SAME
    # components. If one model has calibration and another does not, their
    # composites are weighted differently and are not on a common scale - a
    # silent confound, so it is stated rather than left for the reader to spot.
    component_sets = {tuple(r["components_used"]) for r in ranking}
    uniform = len(component_sets) <= 1
    used = set().union(*component_sets) if component_sets else set()
    components = [k for k in COMPOSITE_WEIGHTS if k in used]
    missing = [k for k in COMPOSITE_WEIGHTS if k not in used]

    if not uniform:
        note = ("WARNING: models were scored over differing component sets, so "
                "their composites use different effective weights and are not "
                "directly comparable. See each row's components_used.")
    elif missing:
        note = (f"composite renormalised over {', '.join(components)}; "
                f"{', '.join(missing)} not measured in this run and excluded "
                f"rather than scored zero")
    else:
        note = "composite uses the full weight set"

    return {
        "benchmark": benchmark,
        "ranking": ranking,
        "comparable": comparable,
        "comparability_note": (
            "all models share one run configuration - differences are attributable "
            "to the models"
            if comparable else
            "WARNING: models were run under differing configurations; the ranking "
            "is confounded. Re-run with a shared RunConfig before citing it."
        ),
        "composite_weights": COMPOSITE_WEIGHTS,
        # Nominal weights above; below, what this particular run applied.
        "components_used": components,
        "components_missing": missing,
        "components_uniform": uniform,
        "effective_weights": (ranking[0]["effective_weights"]
                              if uniform and ranking else None),
        "composite_note": note,
        "components_excluded_by_design": COMPONENTS_EXCLUDED_BY_DESIGN,
        "efficiency_note": (
            "efficiency is relative to the fastest model in THIS run; it is not "
            "comparable across runs, and neither is any composite containing it"
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_probability_inputs(records: Sequence[Dict], n_options: Optional[int]):
    """Pull probability inputs out of per-item records.

    Records without logprobs contribute None, which the calibration layer
    counts as missing coverage rather than as a zero.
    """
    correct_probs, distributions, correctness, correct_idx, ranks = [], [], [], [], []
    any_probs = False

    for r in records:
        correctness.append(bool(r.get("correct")))

        dist = r.get("option_probs") or []
        if dist:
            any_probs = True
        distributions.append(dist)

        cp = r.get("correct_option_prob")
        if cp is not None:
            any_probs = True
        correct_probs.append(cp)

        if n_options:
            letter = r.get("correct_letter")
            correct_idx.append(
                "ABCD".index(letter) if letter in "ABCD" else None
            )
        ranks.append(r.get("target_rank"))

    if not any_probs:
        correct_probs = [None] * len(records)
    if not any(r is not None for r in ranks):
        ranks = None
    return correct_probs, distributions, correctness, (correct_idx or None), ranks


def _comparable_key(config) -> str:
    """Config identity for comparability, ignoring model and run id."""
    if not config:
        return ""
    if hasattr(config, "config_key"):
        return config.config_key()
    import json
    return json.dumps({
        k: config.get(k) for k in ("parallelism", "serving", "decoding", "dataset")
    }, sort_keys=True)


def _bins(run_config) -> int:
    if run_config is None:
        return 10
    ev = getattr(run_config, "evaluation", None)
    if ev is not None:
        return getattr(ev, "calibration_bins", 10)
    if isinstance(run_config, dict):
        return (run_config.get("evaluation") or {}).get("calibration_bins", 10)
    return 10


def _tdp(run_config) -> Optional[float]:
    if run_config is None:
        return None
    sv = getattr(run_config, "serving", None)
    if sv is not None:
        return getattr(sv, "device_tdp_watts", None)
    if isinstance(run_config, dict):
        return (run_config.get("serving") or {}).get("device_tdp_watts")
    return None


def _dig(data: Dict, dotted: str):
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _r(value: Optional[float], places: int = 4) -> Optional[float]:
    return None if value is None else round(value, places)
