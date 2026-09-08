"""Probability and calibration metrics.

Everything here works from token log-probabilities returned by the provider.
OpenRouter only forwards `logprobs` for *some* upstream providers, so every
public function accepts the possibility of missing data and the callers mark
the block as unavailable rather than substituting a guess. A calibration
number computed from absent probabilities would be worse than no number.

All maths is natural-log based (matching the API) and pure stdlib.
"""

import math
from typing import Dict, List, Optional, Sequence

# Smallest probability we are willing to take a log of, so a zero-probability
# observation yields a large-but-finite NLL instead of +inf poisoning a mean.
EPS = 1e-12


# ---------------------------------------------------------------------------
# Basic conversions
# ---------------------------------------------------------------------------

def logprob_to_prob(logprob: Optional[float]) -> Optional[float]:
    """exp() a natural log-probability, tolerating None."""
    if logprob is None:
        return None
    return math.exp(logprob)


def safe_log(p: float) -> float:
    """Natural log clamped away from zero."""
    return math.log(max(p, EPS))


def softmax(scores: Sequence[float]) -> List[float]:
    """Numerically stable softmax over raw scores."""
    if not scores:
        return []
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def normalise(values: Sequence[float]) -> List[float]:
    """Scale non-negative values so they sum to 1. Uniform if all zero."""
    total = sum(values)
    if total <= 0:
        n = len(values)
        return [1.0 / n] * n if n else []
    return [v / total for v in values]


# ---------------------------------------------------------------------------
# Information-theoretic quantities
# ---------------------------------------------------------------------------

def negative_log_likelihood(probs: Sequence[float]) -> Optional[float]:
    """Mean NLL over the probabilities assigned to the *correct* answers.

    Lower is better. This is the quantity language models are trained on, so
    it is the most directly comparable "how surprised was the model" number.
    """
    vals = [p for p in probs if p is not None]
    if not vals:
        return None
    return -sum(safe_log(p) for p in vals) / len(vals)


def perplexity(nll: Optional[float]) -> Optional[float]:
    """exp(NLL). The classic language-model score: 1.0 is perfect."""
    if nll is None:
        return None
    # Guard against overflow on pathologically bad NLL.
    if nll > 700:
        return float("inf")
    return math.exp(nll)


def cross_entropy(probs: Sequence[float]) -> Optional[float]:
    """Cross entropy against the one-hot truth - identical to mean NLL here.

    Kept as a separate name because the report cites both, and readers expect
    to find "cross entropy" as its own row.
    """
    return negative_log_likelihood(probs)


def entropy(distribution: Sequence[float]) -> Optional[float]:
    """Shannon entropy (nats) of a predictive distribution.

    High entropy = the model spread its belief over many options (unsure).
    Zero = all mass on one option (certain, rightly or wrongly).
    """
    vals = [p for p in distribution if p is not None and p > 0]
    if not vals:
        return None
    return -sum(p * math.log(p) for p in vals)


def normalised_entropy(distribution: Sequence[float]) -> Optional[float]:
    """Entropy scaled to 0..1 by the maximum possible for this many options.

    Makes 4-way MMLU entropy comparable with full-vocabulary LAMBADA entropy.
    """
    h = entropy(distribution)
    if h is None:
        return None
    n = len([p for p in distribution if p is not None])
    if n <= 1:
        return 0.0
    return h / math.log(n)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
) -> Optional[Dict]:
    """Expected Calibration Error with equal-width bins.

    Sorts predictions into `n_bins` confidence buckets and compares, per
    bucket, the average confidence against the observed accuracy. ECE is the
    sample-weighted mean of |confidence - accuracy|.

    A model that says "80% sure" and is right 80% of the time has ECE 0.
    Returns the scalar plus the per-bin table, because the table is what makes
    a reliability diagram and is far more informative than the scalar alone.
    """
    pairs = [
        (c, bool(ok))
        for c, ok in zip(confidences, correctness)
        if c is not None
    ]
    if not pairs:
        return None

    n = len(pairs)
    bins = []
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        # Last bin is closed on the right so confidence exactly 1.0 lands in it.
        if i == n_bins - 1:
            members = [p for p in pairs if lo <= p[0] <= hi]
        else:
            members = [p for p in pairs if lo <= p[0] < hi]

        if not members:
            bins.append({
                "bin": i + 1, "range": [round(lo, 3), round(hi, 3)],
                "count": 0, "avg_confidence": None,
                "accuracy": None, "gap": None,
            })
            continue

        avg_conf = sum(c for c, _ in members) / len(members)
        acc = sum(1 for _, ok in members if ok) / len(members)
        gap = abs(avg_conf - acc)
        weight = len(members) / n
        ece += weight * gap
        mce = max(mce, gap)

        bins.append({
            "bin": i + 1,
            "range": [round(lo, 3), round(hi, 3)],
            "count": len(members),
            "avg_confidence": round(avg_conf, 4),
            "accuracy": round(acc, 4),
            "gap": round(gap, 4),
        })

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),          # worst single bin - tail risk
        "n_bins": n_bins,
        "n_samples": n,
        "bins": bins,
    }


def brier_score(
    confidences: Sequence[float],
    correctness: Sequence[bool],
) -> Optional[float]:
    """Mean squared error between confidence and outcome (0 = perfect).

    Unlike ECE this is a *proper scoring rule*: it rewards being both
    well-calibrated and confident, so a model that always says 50% cannot game
    it. Range 0..1 for binary outcomes.
    """
    pairs = [(c, bool(ok)) for c, ok in zip(confidences, correctness) if c is not None]
    if not pairs:
        return None
    return sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in pairs) / len(pairs)


def multiclass_brier_score(
    distributions: Sequence[Sequence[float]],
    correct_indices: Sequence[int],
) -> Optional[float]:
    """Brier score over the full option distribution, not just the top choice.

    For MMLU this is stricter than the binary form: putting 0.4 on the right
    answer and 0.4 on a wrong one is penalised more than 0.4 / 0.2 / 0.2 / 0.2.
    """
    rows = [
        (d, i) for d, i in zip(distributions, correct_indices)
        if d and i is not None and 0 <= i < len(d)
    ]
    if not rows:
        return None
    total = 0.0
    for dist, correct in rows:
        for j, p in enumerate(dist):
            target = 1.0 if j == correct else 0.0
            total += (p - target) ** 2
    return total / len(rows)


def confidence_accuracy_correlation(
    confidences: Sequence[float],
    correctness: Sequence[bool],
) -> Optional[float]:
    """Point-biserial correlation between confidence and being right.

    Answers "does this model *know* when it knows?". Positive is good;
    near zero means its confidence carries no signal, which is a stronger
    indictment than a merely high ECE.
    """
    pairs = [(c, 1.0 if ok else 0.0) for c, ok in zip(confidences, correctness) if c is not None]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    mx = sum(c for c, _ in pairs) / n
    my = sum(y for _, y in pairs) / n
    num = sum((c - mx) * (y - my) for c, y in pairs)
    dx = math.sqrt(sum((c - mx) ** 2 for c, _ in pairs))
    dy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


# ---------------------------------------------------------------------------
# Ranking metrics (for LAMBADA-style open-vocabulary prediction)
# ---------------------------------------------------------------------------

def target_rank(top_logprobs: Sequence[Dict], target_token: str) -> Optional[int]:
    """1-based rank of `target_token` inside the returned top-k list.

    Returns None when the target is outside the top-k window - which is
    itself informative, so callers should count those separately rather than
    dropping them silently.
    """
    if not top_logprobs:
        return None
    norm_target = target_token.strip().lower()
    for i, entry in enumerate(top_logprobs, start=1):
        tok = (entry.get("token") or "").strip().lower()
        if tok == norm_target:
            return i
    return None


def reciprocal_rank(rank: Optional[int]) -> float:
    """1/rank, with 0 for a target that never appeared in the top-k."""
    if rank is None or rank < 1:
        return 0.0
    return 1.0 / rank


def mean_reciprocal_rank(ranks: Sequence[Optional[int]]) -> Optional[float]:
    """MRR over a set of predictions.

    Rewards "nearly right" - a target ranked 2nd scores 0.5 where accuracy
    would score 0. For small models this separates "no idea" from "close".
    """
    if not ranks:
        return None
    return sum(reciprocal_rank(r) for r in ranks) / len(ranks)


def hit_rate_at_k(ranks: Sequence[Optional[int]], k: int) -> Optional[float]:
    """Fraction of targets appearing within the top-k predictions."""
    if not ranks:
        return None
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


# ---------------------------------------------------------------------------
# Aggregation entry point
# ---------------------------------------------------------------------------

def summarise_probability_block(
    correct_probs: Sequence[Optional[float]],
    distributions: Sequence[Sequence[float]],
    correctness: Sequence[bool],
    correct_indices: Optional[Sequence[int]] = None,
    ranks: Optional[Sequence[Optional[int]]] = None,
    n_bins: int = 10,
) -> Dict:
    """Build the full Probability & Calibration block for one model.

    `available` is False when the provider returned no log-probabilities at
    all; downstream renderers show "not exposed by provider" instead of a
    misleading 0.0.
    """
    usable = [p for p in correct_probs if p is not None]
    if not usable:
        return {
            "available": False,
            "reason": "provider did not return token logprobs for this model",
            "coverage": 0.0,
        }

    confidences = [max(d) if d else None for d in distributions]
    entropies = [entropy(d) for d in distributions if d]
    norm_entropies = [normalised_entropy(d) for d in distributions if d]
    nll = negative_log_likelihood(correct_probs)

    block = {
        "available": True,
        "coverage": round(len(usable) / len(correct_probs), 4),
        "correct_option_probability": round(sum(usable) / len(usable), 4),
        "log_probability": round(sum(safe_log(p) for p in usable) / len(usable), 4),
        "nll": round(nll, 4) if nll is not None else None,
        "perplexity": _round_or_inf(perplexity(nll)),
        "cross_entropy": round(nll, 4) if nll is not None else None,
        "brier_score": _round_opt(brier_score(confidences, correctness)),
        "confidence_accuracy_corr": _round_opt(
            confidence_accuracy_correlation(confidences, correctness)
        ),
    }

    if entropies:
        vals = [e for e in entropies if e is not None]
        if vals:
            block["entropy"] = round(sum(vals) / len(vals), 4)
    if norm_entropies:
        vals = [e for e in norm_entropies if e is not None]
        if vals:
            block["normalised_entropy"] = round(sum(vals) / len(vals), 4)

    ece = expected_calibration_error(confidences, correctness, n_bins=n_bins)
    if ece:
        block["ece"] = ece["ece"]
        block["mce"] = ece["mce"]
        block["reliability_bins"] = ece["bins"]

    if correct_indices is not None:
        block["multiclass_brier"] = _round_opt(
            multiclass_brier_score(distributions, correct_indices)
        )

    if ranks is not None:
        block["mean_reciprocal_rank"] = _round_opt(mean_reciprocal_rank(ranks))
        block["hit_rate_at_1"] = _round_opt(hit_rate_at_k(ranks, 1))
        block["hit_rate_at_5"] = _round_opt(hit_rate_at_k(ranks, 5))
        block["hit_rate_at_10"] = _round_opt(hit_rate_at_k(ranks, 10))
        found = [r for r in ranks if r is not None]
        block["mean_target_rank"] = (
            round(sum(found) / len(found), 3) if found else None
        )
        block["target_out_of_topk_rate"] = round(
            sum(1 for r in ranks if r is None) / len(ranks), 4
        )

    return block


def _round_opt(value: Optional[float], places: int = 4) -> Optional[float]:
    return None if value is None else round(value, places)


def _round_or_inf(value: Optional[float], places: int = 4) -> Optional[float]:
    if value is None:
        return None
    if math.isinf(value):
        return None
    return round(value, places)
