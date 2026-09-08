"""Context metrics: is the model actually reading the passage?

LAMBADA is designed so the last word is guessable from the whole passage but
not from the final sentence alone. That design only pays off if we test it,
so this module builds the ablated variants and scores the gap.

The central number is `context_utilization`: accuracy with full context minus
accuracy with the passage cut down. A model scoring 40% on full LAMBADA that
still scores 38% on the last sentence alone is not doing long-range
comprehension - it is exploiting local n-gram statistics, and its headline
number means something quite different from a model that drops to 10%.
"""

import re
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Context ablation generators
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> List[str]:
    """Cheap sentence splitter - adequate for LAMBADA's narrative prose."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def last_sentence_only(context: str) -> str:
    """Keep only the final sentence. The core LAMBADA ablation."""
    sentences = split_sentences(context)
    return sentences[-1] if sentences else context


def last_n_words(context: str, n: int) -> str:
    """Keep only the final n words - a smooth context-length sweep."""
    words = context.split()
    return " ".join(words[-n:]) if len(words) > n else context


def first_half(context: str) -> str:
    """Drop the second half. Should be devastating; if not, something is off."""
    words = context.split()
    return " ".join(words[: max(1, len(words) // 2)])


def shuffled_sentences(context: str, seed: int = 0) -> str:
    """Same sentences, scrambled order.

    Separates "uses the words" from "uses the narrative order". A model
    unaffected by shuffling is doing bag-of-words matching.
    """
    import random
    sentences = split_sentences(context)
    if len(sentences) < 2:
        return context
    head, tail = sentences[:-1], sentences[-1]
    random.Random(seed).shuffle(head)
    return " ".join(head + [tail])


def no_context(context: str) -> str:
    """Empty context - measures the pure unigram prior over target words."""
    return ""


CONTEXT_ABLATIONS = {
    "full": lambda c: c,
    "last_sentence": last_sentence_only,
    "last_20_words": lambda c: last_n_words(c, 20),
    "last_10_words": lambda c: last_n_words(c, 10),
    "first_half": first_half,
    "shuffled_sentences": shuffled_sentences,
    "no_context": no_context,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def context_utilization(
    full_correct: Sequence[bool],
    ablated_correct: Sequence[bool],
) -> Optional[Dict]:
    """Accuracy gap between full context and an ablated version.

    `utilization_ratio` normalises the gap by the full-context accuracy, so it
    reads as "what share of this model's skill depends on the wider passage".
    1.0 = entirely context-driven, 0.0 = context contributed nothing.
    """
    pairs = list(zip(full_correct, ablated_correct))
    if not pairs:
        return None
    n = len(pairs)
    full_acc = sum(1 for f, _ in pairs if f) / n
    abl_acc = sum(1 for _, a in pairs if a) / n

    return {
        "full_accuracy": round(full_acc, 4),
        "ablated_accuracy": round(abl_acc, 4),
        "context_gain": round(full_acc - abl_acc, 4),
        "utilization_ratio": round(
            (full_acc - abl_acc) / full_acc, 4
        ) if full_acc > 0 else None,
        "n_items": n,
    }


def context_length_sensitivity(
    results_by_length: Dict[str, Sequence[bool]],
) -> Optional[Dict]:
    """Accuracy as a function of how much context was supplied.

    A monotonically rising curve is the healthy shape. A flat curve means the
    model ignores the extra context; a falling one means long context actively
    confuses it, which is a real and reportable failure mode for SLMs.
    """
    if not results_by_length:
        return None
    curve = {}
    for label, correctness in results_by_length.items():
        if correctness:
            curve[label] = round(
                sum(1 for c in correctness if c) / len(correctness), 4
            )
    if len(curve) < 2:
        return None

    values = list(curve.values())
    return {
        "curve": curve,
        "span": round(max(values) - min(values), 4),
        "monotonic_increasing": all(
            values[i] <= values[i + 1] for i in range(len(values) - 1)
        ),
    }


def position_sensitivity(
    records: Sequence[Dict],
    n_buckets: int = 4,
) -> Optional[Dict]:
    """Accuracy bucketed by passage length.

    Approximates a "lost in the middle" probe without needing needle
    insertion: if accuracy falls sharply in the longest bucket, the model is
    losing information over distance.
    """
    rows = [
        (len((r.get("context_preview") or r.get("context") or "").split()),
         bool(r.get("correct")))
        for r in records
    ]
    rows = [r for r in rows if r[0] > 0]
    if len(rows) < n_buckets:
        return None

    rows.sort(key=lambda x: x[0])
    size = len(rows) // n_buckets
    buckets = {}
    for i in range(n_buckets):
        start = i * size
        end = (i + 1) * size if i < n_buckets - 1 else len(rows)
        chunk = rows[start:end]
        if not chunk:
            continue
        lengths = [c[0] for c in chunk]
        buckets[f"q{i + 1}"] = {
            "word_range": [min(lengths), max(lengths)],
            "n": len(chunk),
            "accuracy": round(sum(1 for _, ok in chunk if ok) / len(chunk), 4),
        }

    accs = [b["accuracy"] for b in buckets.values()]
    return {
        "buckets": buckets,
        "shortest_vs_longest": round(accs[0] - accs[-1], 4) if len(accs) >= 2 else None,
        "span": round(max(accs) - min(accs), 4) if accs else None,
    }


def build_context_block(
    ablation_results: Optional[Dict[str, Sequence[bool]]] = None,
    records: Optional[Sequence[Dict]] = None,
) -> Dict:
    """Assemble the Context block.

    `ablation_results` maps ablation name -> per-item correctness, and must
    include "full" for the gaps to be computable.
    """
    block: Dict = {"available": False}

    if ablation_results and "full" in ablation_results:
        block["available"] = True
        full = ablation_results["full"]
        ablations = {}
        for name, correctness in ablation_results.items():
            if name == "full":
                continue
            cu = context_utilization(full, correctness)
            if cu:
                ablations[name] = cu
        block["ablations"] = ablations

        if "last_sentence" in ablations:
            # The headline LAMBADA context number.
            block["context_utilization"] = ablations["last_sentence"]["context_gain"]
            block["utilization_ratio"] = ablations["last_sentence"]["utilization_ratio"]
        if "no_context" in ablations:
            block["context_sensitivity"] = ablations["no_context"]["context_gain"]

        length_curve = {
            k: v for k, v in ablation_results.items()
            if k in ("no_context", "last_10_words", "last_20_words",
                     "last_sentence", "full")
        }
        cls = context_length_sensitivity(length_curve)
        if cls:
            block["context_length_sensitivity"] = cls

    if records:
        ps = position_sensitivity(records)
        if ps:
            block["available"] = True
            block["position_sensitivity"] = ps

    if not block["available"]:
        block["reason"] = "context ablation sweep not enabled for this evaluation"
    return block
