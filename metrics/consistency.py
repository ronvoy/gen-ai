"""Consistency metrics: does the model give the same answer twice?

Consistency is orthogonal to accuracy. A model can be consistently wrong
(stable, unhelpful) or accidentally right (unstable, untrustworthy). At
temperature 0 any instability is coming from the serving stack - batching
non-determinism, provider failover, kernel scheduling - not from sampling,
which makes this a systems observation as much as a quality one.
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence


def answer_stability(answer_sets: Sequence[Sequence[str]]) -> Optional[float]:
    """Fraction of items where every repeat produced the identical answer.

    `answer_sets` is one list of repeated answers per item, e.g.
    [["A","A","A"], ["B","C","B"]] -> 0.5.
    """
    rows = [a for a in answer_sets if a]
    if not rows:
        return None
    stable = sum(1 for answers in rows if len(set(answers)) == 1)
    return stable / len(rows)


def majority_vote(answers: Sequence[str]) -> Optional[str]:
    """Most common answer, ties broken by first occurrence."""
    votes = [a for a in answers if a]
    if not votes:
        return None
    counts = Counter(votes)
    best = max(counts.values())
    for a in votes:               # preserve first-seen order on ties
        if counts[a] == best:
            return a
    return None


def self_consistency(
    answer_sets: Sequence[Sequence[str]],
    correct_answers: Sequence[str],
) -> Optional[Dict]:
    """Self-consistency decoding: sample k times, take the majority answer.

    Reports both the majority-vote accuracy and its delta against the mean
    single-sample accuracy. A positive delta means the model has the right
    answer available but does not reliably surface it in one shot.
    """
    rows = [
        (a, c) for a, c in zip(answer_sets, correct_answers) if a
    ]
    if not rows:
        return None

    majority_hits = 0
    single_hits = 0
    single_total = 0
    agreement_scores = []

    for answers, correct in rows:
        if majority_vote(answers) == correct:
            majority_hits += 1
        for a in answers:
            single_total += 1
            single_hits += int(a == correct)
        # How dominant was the winning answer, 1/k .. 1
        counts = Counter(a for a in answers if a)
        if counts:
            agreement_scores.append(max(counts.values()) / len(answers))

    majority_acc = majority_hits / len(rows)
    single_acc = single_hits / single_total if single_total else 0.0

    return {
        "majority_vote_accuracy": round(majority_acc, 4),
        "mean_single_sample_accuracy": round(single_acc, 4),
        "self_consistency_gain": round(majority_acc - single_acc, 4),
        "mean_agreement": round(
            sum(agreement_scores) / len(agreement_scores), 4
        ) if agreement_scores else None,
        "k_samples": len(rows[0][0]) if rows else 0,
        "n_items": len(rows),
    }


def seed_stability(runs_by_seed: Dict[int, Sequence[bool]]) -> Optional[Dict]:
    """Spread of accuracy across independent seeds.

    The headline is `accuracy_std`: if it is comparable to the gap between two
    models, that gap is noise and the report must say so rather than ranking
    them.
    """
    accs = []
    for correctness in runs_by_seed.values():
        if correctness:
            accs.append(sum(1 for c in correctness if c) / len(correctness))
    if len(accs) < 2:
        return None

    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
    std = math.sqrt(var)

    return {
        "n_seeds": len(accs),
        "accuracies": [round(a, 4) for a in accs],
        "mean_accuracy": round(mean, 4),
        "accuracy_std": round(std, 4),
        "accuracy_range": round(max(accs) - min(accs), 4),
        # 95% CI of the mean across seeds
        "ci95_halfwidth": round(1.96 * std / math.sqrt(len(accs)), 4),
    }


def flip_rate(baseline: Sequence[str], variant: Sequence[str]) -> Optional[float]:
    """Share of items whose answer changed between two runs.

    Used both for consistency (same prompt twice) and robustness (perturbed
    prompt). Direction-agnostic: counts any change, right-to-wrong or
    wrong-to-right.
    """
    pairs = [(b, v) for b, v in zip(baseline, variant) if b or v]
    if not pairs:
        return None
    return sum(1 for b, v in pairs if b != v) / len(pairs)


def build_consistency_block(
    answer_sets: Optional[Sequence[Sequence[str]]] = None,
    correct_answers: Optional[Sequence[str]] = None,
    runs_by_seed: Optional[Dict[int, Sequence[bool]]] = None,
) -> Dict:
    """Assemble the Consistency block, marking absent sub-studies explicitly."""
    block: Dict = {"available": False}

    if answer_sets:
        block["available"] = True
        block["answer_stability"] = _r(answer_stability(answer_sets))
        if correct_answers:
            sc = self_consistency(answer_sets, correct_answers)
            if sc:
                block["self_consistency"] = sc

    if runs_by_seed:
        ss = seed_stability(runs_by_seed)
        if ss:
            block["available"] = True
            block["seed_stability"] = ss

    if not block["available"]:
        block["reason"] = "repeat/seed runs not enabled for this evaluation"
    return block


def _r(value: Optional[float], places: int = 4) -> Optional[float]:
    return None if value is None else round(value, places)
