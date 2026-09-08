"""Task-quality metrics for MMLU (multiple choice) and LAMBADA (last word).

These are the metrics that answer "was the model right?", kept strictly
separate from the systems metrics that answer "what did it cost?".

Micro vs macro matters here and is a common source of misreported MMLU
numbers: MMLU subjects have very different sizes, so a plain pooled accuracy
silently over-weights the big subjects. We report both.
"""

import math
import re
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Normalisation shared by both benchmarks
# ---------------------------------------------------------------------------

def normalise_word(word: str) -> str:
    """Lowercase and strip surrounding punctuation for fair comparison."""
    if not word:
        return ""
    return re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", word.lower())


# ---------------------------------------------------------------------------
# Accuracy family
# ---------------------------------------------------------------------------

def overall_accuracy(correctness: Sequence[bool]) -> Optional[float]:
    """Micro accuracy: every question weighs the same."""
    if not correctness:
        return None
    return sum(1 for c in correctness if c) / len(correctness)


def macro_accuracy(group_accuracies: Sequence[float]) -> Optional[float]:
    """Macro accuracy: every *group* weighs the same regardless of size.

    This is the number MMLU papers headline, because it stops a 500-question
    subject from drowning out a 100-question one.
    """
    vals = [a for a in group_accuracies if a is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def grouped_accuracy(
    records: Sequence[Dict],
    group_key: str,
    correct_key: str = "correct",
) -> Dict[str, Dict]:
    """Accuracy broken down by an arbitrary key (subject, category, ...).

    Includes a Wilson 95% confidence interval per group. With 5 questions per
    subject the point estimate alone is close to meaningless, and showing the
    interval is what keeps the report honest about that.
    """
    groups: Dict[str, Dict] = {}
    for rec in records:
        key = rec.get(group_key)
        if key is None:
            continue
        g = groups.setdefault(str(key), {"correct": 0, "total": 0})
        g["total"] += 1
        g["correct"] += int(bool(rec.get(correct_key)))

    for g in groups.values():
        acc = g["correct"] / g["total"] if g["total"] else 0.0
        g["accuracy"] = round(acc, 4)
        lo, hi = wilson_interval(g["correct"], g["total"])
        g["ci95"] = [round(lo, 4), round(hi, 4)]
        g["ci_width"] = round(hi - lo, 4)
    return groups


def error_rate(correctness: Sequence[bool]) -> Optional[float]:
    """1 - accuracy. Reported separately because error budgets read better."""
    acc = overall_accuracy(correctness)
    return None if acc is None else 1.0 - acc


def parse_failure_rate(records: Sequence[Dict], answer_key: str = "predicted_letter") -> Optional[float]:
    """Share of responses we could not extract an answer from at all.

    Distinct from being wrong: an unparseable answer is an *instruction
    following* failure, and for small models it is often the dominant cause of
    a low score. Conflating the two hides the real problem.
    """
    if not records:
        return None
    missing = sum(1 for r in records if not r.get(answer_key))
    return missing / len(records)


def abstention_rate(records: Sequence[Dict], answer_key: str = "predicted_letter") -> Optional[float]:
    """Share of responses where the model declined / returned nothing usable."""
    return parse_failure_rate(records, answer_key)


# ---------------------------------------------------------------------------
# Statistical honesty helpers
# ---------------------------------------------------------------------------

def wilson_interval(successes: int, total: int, z: float = 1.96):
    """Wilson score interval - the right binomial CI for small n.

    The textbook normal approximation breaks down badly at the sample sizes
    used here (5 questions/subject) and can produce intervals outside [0, 1];
    Wilson stays inside and stays sane at n=1.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def random_baseline(n_options: int = 4) -> float:
    """Accuracy a coin-flipper achieves. MMLU's is 25%."""
    return 1.0 / n_options if n_options else 0.0


def normalised_accuracy(accuracy: Optional[float], n_options: int = 4) -> Optional[float]:
    """Rescale accuracy so 0 = random guessing and 1 = perfect.

    ( acc - chance ) / ( 1 - chance ). A 30% MMLU score looks respectable
    until you see it is 0.067 above chance. Negative means below guessing.
    """
    if accuracy is None:
        return None
    chance = random_baseline(n_options)
    if chance >= 1.0:
        return None
    return (accuracy - chance) / (1.0 - chance)


def option_distribution_bias(records: Sequence[Dict], letters=("A", "B", "C", "D")) -> Dict:
    """How often the model picks each letter, vs how often each is correct.

    Small models frequently have a strong positional prior (the notorious
    "always answer C"). Comparing the predicted distribution against the gold
    distribution exposes that, and it explains accuracy that option-reordering
    robustness later destroys.
    """
    pred = {l: 0 for l in letters}
    gold = {l: 0 for l in letters}
    n = 0
    for r in records:
        p = r.get("predicted_letter")
        g = r.get("correct_letter")
        if p in pred:
            pred[p] += 1
        if g in gold:
            gold[g] += 1
        n += 1
    if not n:
        return {}

    pred_share = {l: round(pred[l] / n, 4) for l in letters}
    gold_share = {l: round(gold[l] / n, 4) for l in letters}
    # Total variation distance between the two distributions, 0 = unbiased.
    tvd = 0.5 * sum(abs(pred_share[l] - gold_share[l]) for l in letters)
    return {
        "predicted_share": pred_share,
        "gold_share": gold_share,
        "total_variation_distance": round(tvd, 4),
        "most_favoured_letter": max(pred_share, key=pred_share.get),
    }


# ---------------------------------------------------------------------------
# LAMBADA-specific scoring
# ---------------------------------------------------------------------------

def last_word_accuracy(records: Sequence[Dict]) -> Optional[float]:
    """Standard LAMBADA metric: normalised last word matches exactly."""
    if not records:
        return None
    hits = sum(
        1 for r in records
        if normalise_word(r.get("prediction", "")) == normalise_word(r.get("target", ""))
    )
    return hits / len(records)


def exact_target_match(records: Sequence[Dict]) -> Optional[float]:
    """Stricter: byte-identical match including case and punctuation.

    The gap between this and `last_word_accuracy` measures how much of the
    score depends on our normalisation being generous.
    """
    if not records:
        return None
    hits = sum(
        1 for r in records
        if (r.get("prediction") or "") == (r.get("target") or "")
    )
    return hits / len(records)


def stem_match_accuracy(records: Sequence[Dict]) -> Optional[float]:
    """Credit predictions that differ only by a trailing inflection.

    "run"/"running", "dog"/"dogs". Not a standard LAMBADA metric, but it
    separates "wrong word" from "right word, wrong form", which is a
    meaningfully different failure for a small model.
    """
    if not records:
        return None

    def stem(w: str) -> str:
        w = normalise_word(w)
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > len(suf) + 2 and w.endswith(suf):
                return w[: -len(suf)]
        return w

    hits = sum(
        1 for r in records
        if stem(r.get("prediction", "")) == stem(r.get("target", ""))
    )
    return hits / len(records)


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------

def build_mmlu_quality_block(records: Sequence[Dict]) -> Dict:
    """The complete MMLU Task Quality block."""
    correctness = [bool(r.get("correct")) for r in records]
    subjects = grouped_accuracy(records, "subject")
    categories = grouped_accuracy(records, "category")

    acc = overall_accuracy(correctness)
    lo, hi = wilson_interval(sum(correctness), len(correctness)) if records else (0, 0)

    return {
        "overall_accuracy": _r(acc),
        "overall_accuracy_ci95": [round(lo, 4), round(hi, 4)],
        "macro_accuracy_subject": _r(
            macro_accuracy([s["accuracy"] for s in subjects.values()])
        ),
        "macro_accuracy_category": _r(
            macro_accuracy([c["accuracy"] for c in categories.values()])
        ),
        "normalised_accuracy": _r(normalised_accuracy(acc, 4)),
        "random_baseline": 0.25,
        "error_rate": _r(error_rate(correctness)),
        "parse_failure_rate": _r(parse_failure_rate(records)),
        "subject_accuracy": subjects,
        "category_accuracy": categories,
        "option_bias": option_distribution_bias(records),
        "n_questions": len(records),
        "n_subjects": len(subjects),
    }


def build_lambada_quality_block(records: Sequence[Dict]) -> Dict:
    """The complete LAMBADA Task Quality block."""
    correctness = [
        normalise_word(r.get("prediction", "")) == normalise_word(r.get("target", ""))
        for r in records
    ]
    acc = overall_accuracy(correctness)
    lo, hi = wilson_interval(sum(correctness), len(correctness)) if records else (0, 0)

    return {
        "last_word_accuracy": _r(last_word_accuracy(records)),
        "last_word_accuracy_ci95": [round(lo, 4), round(hi, 4)],
        "exact_target_match": _r(exact_target_match(records)),
        "stem_match_accuracy": _r(stem_match_accuracy(records)),
        "error_rate": _r(error_rate(correctness)),
        "empty_prediction_rate": _r(
            parse_failure_rate(records, answer_key="prediction")
        ),
        "n_passages": len(records),
    }


def _r(value: Optional[float], places: int = 4) -> Optional[float]:
    return None if value is None else round(value, places)
