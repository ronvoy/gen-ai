"""Significance and variance analysis over per-item benchmark results.

An accuracy table says Ministral scored 79.3% and Gemma 60.4%. It does not say
whether that gap could be sampling noise, nor which of the things we varied
actually moves the outcome. This module answers both from the per-item records
the runners already write.

Pure standard library, deliberately: the report has to be regenerable on a
shared host where scipy cannot be installed. `tests/test_significance.py`
cross-checks every statistic here against scipy/statsmodels.

Tests provided
--------------
McNemar          paired, two models on the same items. The right test here:
                 the models answered identical questions, so an unpaired
                 two-proportion test would throw away the pairing and overstate
                 the variance.
Cochran's Q      the k-model omnibus. Run first, so that pairwise tests are
                 only interpreted when something differs at all.
Holm-Bonferroni  three pairwise comparisons means three chances at a false
                 positive; Holm controls that without Bonferroni's power loss.
Chi-square       independence of correctness from a factor (subject category,
                 target fragmentation, passage length) - the variance question.
Variance
decomposition    how much of the spread in per-subject accuracy is real signal
                 and how much is binomial noise from ~50 items per subject.
"""

import math
from typing import Dict, List, Sequence

# ---------------------------------------------------------------------------
# Distribution functions
#
# Implemented rather than imported so this module stays stdlib-only. Both are
# the standard Numerical Recipes formulations; the test module pins them
# against scipy to 1e-10.
# ---------------------------------------------------------------------------

_MAX_ITER = 500
_EPS = 3e-16


def _gamma_p_series(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a,x), by series. Good for x < a+1."""
    ap, total, term = a, 1.0 / a, 1.0 / a
    for _ in range(_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_cf(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a,x), by continued fraction."""
    tiny = 1e-300
    b, c, d = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, _MAX_ITER):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(x: float, df: int) -> float:
    """P(X > x) for a chi-square with `df` degrees of freedom."""
    if df <= 0:
        return float("nan")
    if x <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    return _gamma_q_cf(a, xx) if xx >= a + 1.0 else 1.0 - _gamma_p_series(a, xx)


def binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test. At p=0.5 this is the sign test."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


# ---------------------------------------------------------------------------
# Paired model comparison
# ---------------------------------------------------------------------------

def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool],
            exact_threshold: int = 25) -> Dict:
    """McNemar's test for two models scored on the same items, in order.

    Only the discordant pairs carry information: items both models got right,
    or both wrong, say nothing about which is better. `b` counts items the
    first model got right and the second wrong, `c` the reverse.

    Below `exact_threshold` discordant pairs the chi-square approximation is
    unreliable, so an exact binomial test is used instead.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("paired test needs equal-length, item-aligned inputs")

    n = len(a_correct)
    both = only_a = only_b = neither = 0
    for x, y in zip(a_correct, b_correct):
        if x and y:
            both += 1
        elif x:
            only_a += 1
        elif y:
            only_b += 1
        else:
            neither += 1

    b, c = only_a, only_b
    discordant = b + c

    if discordant == 0:
        return {
            "n": n, "both_correct": both, "both_wrong": neither,
            "only_a_correct": b, "only_b_correct": c, "discordant": 0,
            "statistic": None, "p_value": 1.0, "test": "none",
            "accuracy_delta": 0.0, "delta_ci95": [0.0, 0.0],
            "odds_ratio": None, "significant": False,
            "note": "the two models were correct on exactly the same items",
        }

    if discordant < exact_threshold:
        test = "exact binomial"
        stat = None
        p = binom_two_sided_p(b, discordant)
    else:
        # Continuity-corrected chi-square, 1 df.
        test = "chi-square (continuity-corrected)"
        stat = (abs(b - c) - 1.0) ** 2 / discordant
        p = chi2_sf(stat, 1)

    # Paired difference in accuracy, with the McNemar standard error. The
    # pairing is what makes this interval much tighter than an unpaired one.
    delta = (b - c) / n
    se = math.sqrt(discordant - (b - c) ** 2 / n) / n if n else 0.0
    half = 1.959963985 * se

    return {
        "n": n,
        "both_correct": both, "both_wrong": neither,
        "only_a_correct": b, "only_b_correct": c,
        "discordant": discordant,
        "statistic": round(stat, 4) if stat is not None else None,
        "p_value": p,
        "test": test,
        "accuracy_delta": round(delta, 4),
        "delta_ci95": [round(delta - half, 4), round(delta + half, 4)],
        "odds_ratio": round(b / c, 3) if c else None,
        "significant": p < 0.05,
    }


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> List[Dict]:
    """Holm's step-down correction over a family of p-values.

    Three pairwise comparisons give three chances at a false positive. Holm
    controls the family-wise error rate exactly as Bonferroni does but rejects
    at least as much, so it costs nothing to prefer it.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    out = [None] * m
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p_values[idx])
        running = max(running, adjusted)  # enforce monotonicity
        out[idx] = {
            "p_raw": p_values[idx],
            "p_adjusted": running,
            "significant": running < alpha,
        }
    return out


def pairwise_mcnemar(models: Dict[str, Sequence[bool]]) -> List[Dict]:
    """Every pair, McNemar-tested, with Holm correction across the family."""
    names = list(models)
    pairs = [(names[i], names[j])
             for i in range(len(names)) for j in range(i + 1, len(names))]
    results = [dict(mcnemar(models[a], models[b]), model_a=a, model_b=b)
               for a, b in pairs]
    for r, adj in zip(results, holm_bonferroni([r["p_value"] for r in results])):
        r["p_adjusted"] = adj["p_adjusted"]
        r["significant_adjusted"] = adj["significant"]
    return results


def cochrans_q(models: Dict[str, Sequence[bool]]) -> Dict:
    """Cochran's Q: do k models differ at all, on the same items?

    The omnibus that licenses the pairwise tests. Without it, running three
    comparisons and reporting the smallest p-value would be fishing.
    """
    names = list(models)
    k = len(names)
    if k < 2:
        return {"available": False, "reason": "needs at least two models"}
    n = len(models[names[0]])
    if any(len(models[m]) != n for m in names):
        raise ValueError("Cochran's Q needs item-aligned inputs")

    col = [sum(1 for v in models[m] if v) for m in names]
    row = [sum(1 for m in names if models[m][i]) for i in range(n)]

    denom = k * sum(row) - sum(r * r for r in row)
    if denom == 0:
        return {"available": False,
                "reason": "every item was answered identically by all models"}

    q = (k - 1) * (k * sum(g * g for g in col) - sum(col) ** 2) / denom
    df = k - 1
    p = chi2_sf(q, df)
    return {
        "available": True, "k_models": k, "n_items": n,
        "q_statistic": round(q, 3), "df": df, "p_value": p,
        "significant": p < 0.05,
        "per_model_correct": dict(zip(names, col)),
    }


# ---------------------------------------------------------------------------
# Which parameters move the outcome
# ---------------------------------------------------------------------------

def chi2_independence(groups: Dict[str, Sequence[bool]]) -> Dict:
    """Is correctness independent of the group an item falls in?

    `groups` maps a factor level (a subject category, a fragmentation bucket, a
    length quartile) to the correctness values of the items at that level.
    Cramer's V accompanies the p-value because with thousands of items a
    trivial association is significant; V says whether it is also large.
    """
    levels = [g for g, v in groups.items() if len(v) > 0]
    if len(levels) < 2:
        return {"available": False, "reason": "needs at least two levels"}

    obs = [[sum(1 for v in groups[g] if v), sum(1 for v in groups[g] if not v)]
           for g in levels]
    n = sum(sum(r) for r in obs)
    col_tot = [sum(r[j] for r in obs) for j in (0, 1)]
    if not n or 0 in col_tot:
        return {"available": False,
                "reason": "every item shares one outcome; nothing to test"}

    chi2 = 0.0
    for row in obs:
        rt = sum(row)
        for j in (0, 1):
            exp = rt * col_tot[j] / n
            if exp > 0:
                chi2 += (row[j] - exp) ** 2 / exp

    df = len(levels) - 1
    p = chi2_sf(chi2, df)
    # Cramer's V; with 2 columns min(r-1, c-1) is 1, so V = sqrt(chi2/n).
    v = math.sqrt(chi2 / n)
    return {
        "available": True,
        "levels": levels,
        "n": n,
        "accuracy_by_level": {g: round(sum(1 for x in groups[g] if x) / len(groups[g]), 4)
                              for g in levels},
        "n_by_level": {g: len(groups[g]) for g in levels},
        "chi2": round(chi2, 3), "df": df, "p_value": p,
        "cramers_v": round(v, 4),
        "effect": ("negligible" if v < 0.1 else "small" if v < 0.3
                   else "moderate" if v < 0.5 else "large"),
        "significant": p < 0.05,
        "spread": round(max(sum(1 for x in groups[g] if x) / len(groups[g]) for g in levels)
                        - min(sum(1 for x in groups[g] if x) / len(groups[g]) for g in levels), 4),
    }


def variance_decomposition(group_scores: Dict[str, Sequence[bool]]) -> Dict:
    """Split the spread in per-group accuracy into real signal and noise.

    MMLU reports 57 subject accuracies from ~50 questions each. Some of that
    spread is genuine subject difficulty; some is just what 50 Bernoulli draws
    do. Subtracting the expected binomial variance from the observed variance
    leaves the part attributable to the subject itself.

    `between_share` near 0 means the per-subject ranking is mostly noise and
    should not be read; near 1 means subject difficulty is real and large.

    One caveat worth carrying into the write-up: the estimator is truncated at
    zero, so when the true between-group variance really is zero, sampling
    error can only push the estimate up, never down. Under a null of identical
    subjects and 57 groups of 50, `between_share` still averages around 0.08
    and occasionally reaches 0.25 - see tests/test_significance.py. Read a
    small share as "indistinguishable from noise", not as a measured quantity.
    """
    stats = [(g, len(v), sum(1 for x in v if x) / len(v))
             for g, v in group_scores.items() if len(v) > 1]
    if len(stats) < 2:
        return {"available": False, "reason": "needs at least two groups"}

    accs = [p for _, _, p in stats]
    mean = sum(accs) / len(accs)
    observed = sum((p - mean) ** 2 for p in accs) / (len(accs) - 1)
    # Expected variance from finite sampling alone, averaged over groups.
    within = sum(p * (1 - p) / n for _, n, p in stats) / len(stats)
    between = max(0.0, observed - within)

    ranked = sorted(stats, key=lambda t: t[2])
    return {
        "available": True,
        "n_groups": len(stats),
        "mean_accuracy": round(mean, 4),
        "observed_variance": round(observed, 6),
        "sampling_variance": round(within, 6),
        "between_group_variance": round(between, 6),
        "between_share": round(between / observed, 4) if observed else 0.0,
        "observed_sd": round(math.sqrt(observed), 4),
        "true_sd": round(math.sqrt(between), 4),
        "hardest": [(g, round(p, 3)) for g, _, p in ranked[:3]],
        "easiest": [(g, round(p, 3)) for g, _, p in ranked[-3:][::-1]],
    }


def error_overlap(models: Dict[str, Sequence[bool]]) -> List[Dict]:
    """Do the models fail on the same items?

    Highly overlapping failures mean the items are hard; disjoint failures mean
    the models have different competences and an ensemble would gain. Phi is
    the correlation between the two correctness vectors.
    """
    names = list(models)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = models[names[i]], models[names[j]]
            n = len(a)
            both_wrong = sum(1 for x, y in zip(a, b) if not x and not y)
            either_wrong = sum(1 for x, y in zip(a, b) if not x or not y)
            n11 = sum(1 for x, y in zip(a, b) if x and y)
            n10 = sum(1 for x, y in zip(a, b) if x and not y)
            n01 = sum(1 for x, y in zip(a, b) if not x and y)
            n00 = both_wrong
            den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
            phi = ((n11 * n00 - n10 * n01) / den) if den else None
            out.append({
                "model_a": names[i], "model_b": names[j],
                "agreement": round(sum(1 for x, y in zip(a, b) if x == y) / n, 4),
                "shared_failure_rate": round(both_wrong / n, 4),
                "jaccard_failures": round(both_wrong / either_wrong, 4) if either_wrong else None,
                "phi": round(phi, 4) if phi is not None else None,
            })
    return out


def oracle_ceiling(models: Dict[str, Sequence[bool]]) -> Dict:
    """What an ideal router over these models could reach.

    The gap between the best single model and the any-model-correct ceiling is
    the headroom that model selection could buy, and it is invisible in a table
    of individual accuracies.
    """
    names = list(models)
    n = len(models[names[0]])
    best = max((sum(1 for v in models[m] if v) / n, m) for m in names)
    any_correct = sum(1 for i in range(n) if any(models[m][i] for m in names)) / n
    all_correct = sum(1 for i in range(n) if all(models[m][i] for m in names)) / n
    return {
        "best_single_model": best[1],
        "best_single_accuracy": round(best[0], 4),
        "oracle_accuracy": round(any_correct, 4),
        "headroom": round(any_correct - best[0], 4),
        "all_models_correct": round(all_correct, 4),
        "none_correct": round(1 - any_correct, 4),
    }


def format_p(p) -> str:
    """Consistent p-value rendering for the report tables."""
    if p is None:
        return "-"
    if p < 1e-4:
        return "< 0.0001"
    return f"{p:.4f}"
