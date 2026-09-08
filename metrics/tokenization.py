"""Tokenization metrics.

LAMBADA asks for one *word*, but models predict *tokens*. A target that the
tokenizer splits into three pieces has to be got right three times over, so
fragmentation is a direct, mechanical handicap that has nothing to do with
whether the model understood the passage. Measuring it separates tokenizer
disadvantage from comprehension failure - without it, comparing models with
different vocabularies is not a fair fight.

Tokenizers are loaded lazily and cached; when none is reachable (offline, or
a gated repo) we fall back to a documented heuristic and set
`tokenizer_exact: False` so the report never presents an estimate as measured.
"""

import os
import re
from typing import Dict, List, Optional, Sequence

# HF repo ids for the tokenizers matching the benchmarked models. These are
# small (a few MB) and cached under ~/.cache/huggingface after first use.
TOKENIZER_REPOS = {
    "google/gemma-3-4b-it": "google/gemma-3-4b-it",
    "meta-llama/llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/ministral-8b-2512": "mistralai/Ministral-8B-Instruct-2410",
}

# Generic stand-in when the exact tokenizer is gated or unavailable offline.
FALLBACK_ENCODING = "cl100k_base"

_CACHE: Dict[str, object] = {}
_FAILED: set = set()


def get_tokenizer(model: str):
    """Return a callable text -> list[str] of tokens, or None.

    Tries the model's own HF tokenizer first (exact), then tiktoken
    (approximate but stable), then gives up.
    """
    if model in _CACHE:
        return _CACHE[model]
    if model in _FAILED:
        return None

    repo = TOKENIZER_REPOS.get(model)
    if repo and not os.getenv("SLM_BENCH_NO_HF_TOKENIZER"):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(repo)

            def _encode(text, _tok=tok):
                return _tok.tokenize(text)

            _encode.exact = True
            _encode.name = repo
            _CACHE[model] = _encode
            return _encode
        except Exception:
            pass

    try:
        import tiktoken
        enc = tiktoken.get_encoding(FALLBACK_ENCODING)

        def _encode_tt(text, _enc=enc):
            return [
                _enc.decode_single_token_bytes(t).decode("utf-8", "replace")
                for t in _enc.encode(text)
            ]

        _encode_tt.exact = False
        _encode_tt.name = FALLBACK_ENCODING
        _CACHE[model] = _encode_tt
        return _encode_tt
    except Exception:
        _FAILED.add(model)
        return None


def heuristic_token_count(word: str) -> int:
    """Rough token count when no tokenizer is available.

    Byte-pair vocabularies average ~4 characters per token on English text;
    this is only used to keep the pipeline running, never reported as exact.
    """
    if not word:
        return 0
    return max(1, round(len(word) / 4))


def tokens_for_word(word: str, tokenizer) -> int:
    """Token count for a single target word, with a leading space.

    The leading space matters: BPE vocabularies encode " dog" and "dog" as
    different tokens, and LAMBADA targets always follow a space in context.
    Omitting it inflates the count and would make every model look worse.
    """
    if not word:
        return 0
    if tokenizer is None:
        return heuristic_token_count(word)
    try:
        return len(tokenizer(" " + word))
    except Exception:
        return heuristic_token_count(word)


def tokens_per_target_word(
    targets: Sequence[str], tokenizer
) -> Optional[float]:
    """Mean number of tokens the tokenizer needs per gold target word."""
    counts = [tokens_for_word(t, tokenizer) for t in targets if t]
    if not counts:
        return None
    return sum(counts) / len(counts)


def fragmentation_rate(targets: Sequence[str], tokenizer) -> Optional[float]:
    """Share of targets that are NOT a single token.

    This is the fraction of the benchmark where the model must produce a
    multi-token continuation correctly - strictly harder than a single-token
    lookup, and unequal across model families.
    """
    counts = [tokens_for_word(t, tokenizer) for t in targets if t]
    if not counts:
        return None
    return sum(1 for c in counts if c > 1) / len(counts)


def subword_exact_match(records: Sequence[Dict], tokenizer) -> Optional[float]:
    """Match at the token-sequence level rather than the string level.

    Catches cases where the prediction renders identically but tokenizes
    differently (leading space, unicode normalisation). Divergence between
    this and string accuracy points at a text-handling bug, not a model one.
    """
    if not records:
        return None
    hits = 0
    for r in records:
        pred = (r.get("prediction") or "").strip()
        target = (r.get("target") or "").strip()
        if not pred or not target:
            continue
        if tokenizer is None:
            hits += int(pred.lower() == target.lower())
            continue
        try:
            hits += int(tokenizer(" " + pred) == tokenizer(" " + target))
        except Exception:
            hits += int(pred.lower() == target.lower())
    return hits / len(records)


def accuracy_by_fragmentation(
    records: Sequence[Dict], tokenizer
) -> Dict[str, Dict]:
    """Accuracy split by how many tokens the target needs.

    The payoff metric of this module: if accuracy collapses from the 1-token
    bucket to the 3+-token bucket, the model's LAMBADA score is substantially
    a tokenizer artefact and the report should say so.
    """
    buckets: Dict[str, Dict] = {}
    for r in records:
        target = r.get("target") or ""
        if not target:
            continue
        n = tokens_for_word(target, tokenizer)
        key = "1_token" if n == 1 else ("2_tokens" if n == 2 else "3plus_tokens")
        b = buckets.setdefault(key, {"correct": 0, "total": 0})
        b["total"] += 1
        pred = _norm(r.get("prediction", ""))
        b["correct"] += int(pred == _norm(target))

    for b in buckets.values():
        b["accuracy"] = round(b["correct"] / b["total"], 4) if b["total"] else None
    return buckets


def build_tokenization_block(records: Sequence[Dict], model: str) -> Dict:
    """Assemble the Tokenization block for one model."""
    tokenizer = get_tokenizer(model)
    targets = [r.get("target", "") for r in records if r.get("target")]

    if not targets:
        return {"available": False, "reason": "no target words in results"}

    return {
        "available": True,
        "tokenizer": getattr(tokenizer, "name", "heuristic"),
        "tokenizer_exact": bool(getattr(tokenizer, "exact", False)),
        "tokens_per_target_word": _r(tokens_per_target_word(targets, tokenizer), 3),
        "fragmentation_rate": _r(fragmentation_rate(targets, tokenizer)),
        "subword_exact_match": _r(subword_exact_match(records, tokenizer)),
        "accuracy_by_fragmentation": accuracy_by_fragmentation(records, tokenizer),
    }


def _norm(word: str) -> str:
    return re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", (word or "").lower())


def _r(value: Optional[float], places: int = 4) -> Optional[float]:
    return None if value is None else round(value, places)
