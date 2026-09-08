"""Robustness: does the score survive a harmless change to the input?

Two halves:
  * generators - deterministic, seeded input perturbations, so a robustness
    run is reproducible and the exact perturbed prompt is recoverable;
  * scorers    - accuracy deltas and flip rates against the clean baseline.

Design note: every perturbation here is *meaning-preserving*. A change that
alters the correct answer would measure dataset noise, not model robustness.
Option reordering is the one that moves the gold letter, so it carries the
remapping with it.
"""

import random
import re
import string
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Prompt-template variation (MMLU)
# ---------------------------------------------------------------------------

# Semantically identical instructions in different wordings. A model whose
# accuracy swings across these is over-fitted to one phrasing.
PROMPT_TEMPLATES = {
    "baseline": (
        "The following is a multiple choice question about {subject}.\n"
        "Think step by step, then give your final answer as a single letter.\n\n"
        "{question}\n{options}\n\nReasoning:"
    ),
    "terse": (
        "{subject} question.\n\n{question}\n{options}\n\n"
        "Answer with one letter (A, B, C or D):"
    ),
    "verbose": (
        "You are an expert examiner in the field of {subject}. Below you will "
        "find a multiple choice question with four candidate answers. Consider "
        "each option carefully, explain your reasoning, and then state which "
        "single option is correct.\n\n{question}\n{options}\n\nYour analysis:"
    ),
    "role_free": (
        "Question ({subject}):\n{question}\n{options}\n\n"
        "Which option is correct? Explain briefly, then answer:"
    ),
    "answer_first": (
        "{subject}\n\n{question}\n{options}\n\n"
        "State the correct letter first, then justify it:"
    ),
}


# ---------------------------------------------------------------------------
# Character / token level noise (both benchmarks)
# ---------------------------------------------------------------------------

KEYBOARD_NEIGHBOURS = {
    "a": "qws", "b": "vgn", "c": "xdv", "d": "sfe", "e": "wrd", "f": "dgr",
    "g": "fht", "h": "gjy", "i": "uok", "j": "hkn", "k": "jlm", "l": "kop",
    "m": "njk", "n": "bmh", "o": "ipl", "p": "ol", "q": "wa", "r": "etf",
    "s": "adw", "t": "ryg", "u": "yij", "v": "cbf", "w": "qes", "x": "zsc",
    "y": "tuh", "z": "asx",
}


def typo_perturb(text: str, rate: float = 0.05, seed: int = 0) -> str:
    """Realistic typing errors: adjacent-key substitutions and transpositions.

    Only touches word interiors so the first/last letter stays put - that is
    how human typos actually distribute, and it keeps words recognisable.
    """
    rng = random.Random(seed)
    words = text.split(" ")
    out = []
    for w in words:
        if len(w) < 4 or rng.random() > rate:
            out.append(w)
            continue
        i = rng.randrange(1, len(w) - 1)
        ch = w[i].lower()
        if rng.random() < 0.5 and ch in KEYBOARD_NEIGHBOURS:
            repl = rng.choice(KEYBOARD_NEIGHBOURS[ch])
            out.append(w[:i] + repl + w[i + 1:])
        else:                                   # transpose with next char
            out.append(w[:i] + w[i + 1] + w[i] + w[i + 2:])
    return " ".join(out)


def whitespace_noise(text: str, rate: float = 0.05, seed: int = 0) -> str:
    """Insert stray double spaces - tests tokenizer brittleness, not meaning."""
    rng = random.Random(seed)
    words = text.split(" ")
    return " ".join(w + " " if rng.random() < rate else w for w in words)


def casing_noise(text: str, rate: float = 0.08, seed: int = 0) -> str:
    """Randomly upper-case whole words, as in sloppy user input."""
    rng = random.Random(seed)
    return " ".join(
        w.upper() if rng.random() < rate else w for w in text.split(" ")
    )


def punctuation_noise(text: str, rate: float = 0.05, seed: int = 0) -> str:
    """Drop or duplicate punctuation marks."""
    rng = random.Random(seed)
    out = []
    for ch in text:
        if ch in string.punctuation and rng.random() < rate:
            if rng.random() < 0.5:
                continue                        # drop it
            out.append(ch)                      # duplicate it
        out.append(ch)
    return "".join(out)


def unicode_lookalike(text: str, rate: float = 0.03, seed: int = 0) -> str:
    """Swap ASCII for visually identical Unicode (homoglyph attack).

    Included because it cleanly separates "reads the words" from "matches the
    byte pattern": a human sees no change at all, a tokenizer sees a different
    token sequence entirely.
    """
    homoglyphs = {"a": "а", "e": "е", "o": "о",
                  "p": "р", "c": "с", "x": "х"}
    rng = random.Random(seed)
    return "".join(
        homoglyphs.get(ch, ch) if ch in homoglyphs and rng.random() < rate else ch
        for ch in text
    )


PERTURBATIONS: Dict[str, Callable[..., str]] = {
    "typo": typo_perturb,
    "whitespace": whitespace_noise,
    "casing": casing_noise,
    "punctuation": punctuation_noise,
    "unicode_lookalike": unicode_lookalike,
}


# ---------------------------------------------------------------------------
# Option reordering (MMLU)
# ---------------------------------------------------------------------------

def reorder_options(
    choices: Sequence[str],
    correct_index: int,
    strategy: str = "reverse",
    seed: int = 0,
) -> Tuple[List[str], int]:
    """Permute answer options and return the new gold index alongside.

    Strategies:
      reverse       - deterministic, maximal positional displacement
      shuffle       - seeded random permutation
      move_to_last  - forces the answer into position D (worst case for a
                      model with an early-position prior)
      move_to_first - forces it into position A
    """
    n = len(choices)
    order = list(range(n))

    if strategy == "reverse":
        order = order[::-1]
    elif strategy == "shuffle":
        random.Random(seed).shuffle(order)
    elif strategy == "move_to_last":
        order.remove(correct_index)
        order.append(correct_index)
    elif strategy == "move_to_first":
        order.remove(correct_index)
        order.insert(0, correct_index)

    new_choices = [choices[i] for i in order]
    new_correct = order.index(correct_index)
    return new_choices, new_correct


# ---------------------------------------------------------------------------
# Paraphrase (light, dependency-free)
# ---------------------------------------------------------------------------

# A learned paraphraser would need another model in the loop and would make
# the benchmark non-reproducible. These deterministic rewrites are weaker but
# auditable - the report states this limitation explicitly.
PARAPHRASE_RULES = [
    (r"\bWhich of the following\b", "Which one"),
    (r"\bwhich of the following\b", "which one"),
    (r"\bis the most likely\b", "is most probably"),
    (r"\bWhat is\b", "What would be"),
    (r"\bAccording to\b", "Based on"),
    (r"\bare the\b", "would be the"),
    (r"\bcan be\b", "may be"),
    (r"\bmust be\b", "has to be"),
]


def paraphrase(text: str) -> str:
    """Apply deterministic surface rewrites that preserve meaning."""
    out = text
    for pattern, repl in PARAPHRASE_RULES:
        out = re.sub(pattern, repl, out)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def robustness_delta(
    baseline_correct: Sequence[bool],
    variant_correct: Sequence[bool],
) -> Optional[Dict]:
    """Compare a perturbed run against the clean baseline.

    `accuracy_drop` is the headline. `flip_rate` catches the case where the
    accuracy is unchanged because equal numbers of answers broke and fixed
    themselves - which is instability masquerading as robustness, and the
    reason accuracy alone is not enough here.
    """
    pairs = list(zip(baseline_correct, variant_correct))
    if not pairs:
        return None

    n = len(pairs)
    base_acc = sum(1 for b, _ in pairs if b) / n
    var_acc = sum(1 for _, v in pairs if v) / n
    broke = sum(1 for b, v in pairs if b and not v)
    fixed = sum(1 for b, v in pairs if not b and v)

    return {
        "baseline_accuracy": round(base_acc, 4),
        "variant_accuracy": round(var_acc, 4),
        "accuracy_drop": round(base_acc - var_acc, 4),
        "relative_drop": round(
            (base_acc - var_acc) / base_acc, 4
        ) if base_acc > 0 else None,
        "broke": broke,          # was right, now wrong
        "fixed": fixed,          # was wrong, now right
        "flip_rate": round((broke + fixed) / n, 4),
        "n_items": n,
    }


def robustness_score(deltas: Sequence[Optional[Dict]]) -> Optional[float]:
    """Single 0..1 robustness figure: 1 - mean relative accuracy drop.

    Clipped at 0 so a catastrophic perturbation cannot drag a model negative
    and distort the composite. Higher is better.
    """
    drops = [
        d["relative_drop"] for d in deltas
        if d and d.get("relative_drop") is not None
    ]
    if not drops:
        return None
    mean_drop = sum(drops) / len(drops)
    return max(0.0, min(1.0, 1.0 - mean_drop))


def build_robustness_block(variant_results: Dict[str, Dict]) -> Dict:
    """Assemble the Robustness block from {variant_name: robustness_delta}."""
    if not variant_results:
        return {
            "available": False,
            "reason": "robustness sweep not enabled for this evaluation",
        }
    deltas = list(variant_results.values())
    worst = None
    worst_name = None
    for name, d in variant_results.items():
        if d and d.get("accuracy_drop") is not None:
            if worst is None or d["accuracy_drop"] > worst:
                worst, worst_name = d["accuracy_drop"], name

    return {
        "available": True,
        "variants": variant_results,
        "robustness_score": _r(robustness_score(deltas)),
        "worst_variant": worst_name,
        "worst_accuracy_drop": _r(worst),
        "mean_flip_rate": _r(_mean(
            [d["flip_rate"] for d in deltas if d and d.get("flip_rate") is not None]
        )),
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _r(value: Optional[float], places: int = 4) -> Optional[float]:
    return None if value is None else round(value, places)
