"""Instrumented OpenRouter client: latency telemetry + log-probabilities.

Two things the plain client cannot give us:

  TTFT   requires streaming. Without it we only see end-to-end time, which
         conflates prefill and decode and makes TPOT uncomputable.
  logprobs  requires asking for them - and then coping with the fact that
         OpenRouter only forwards them for *some* upstream providers.

That second point is measured, not assumed: as of writing, Llama-3.2-3B via
Parasail returns logprobs while Gemma-3-4B via DeepInfra and Ministral-8B via
Mistral do not. `probe_capabilities()` checks at runtime so the report states
what was actually available on the day, and the calibration block is marked
unavailable rather than silently filled with nulls-as-zeros.
"""

import json
import time
from typing import Callable, Dict, List, Optional

import requests

from config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_MAX_RETRIES,
    OPENROUTER_PROVIDER_SORT,
    OPENROUTER_RETRY_BASE_DELAY,
    OPENROUTER_RETRY_MAX_DELAY,
)

RETRYABLE = {429, 500, 502, 503, 504}


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/slm-benchmark",
        "X-Title": "SLM MMLU/LAMBADA Benchmark",
    }


def _sleep_for(attempt: int, resp: Optional[requests.Response]) -> float:
    import random
    if resp is not None:
        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                return float(ra) + random.uniform(0, 0.5)
            except ValueError:
                pass
    return min(
        OPENROUTER_RETRY_BASE_DELAY * (2 ** (attempt - 1)),
        OPENROUTER_RETRY_MAX_DELAY,
    ) + random.uniform(0, 0.5)


# ---------------------------------------------------------------------------
# Streaming call with telemetry
# ---------------------------------------------------------------------------

def stream_completion(
    model: str,
    prompt: str,
    api_key: str,
    params: Optional[Dict] = None,
    request_logprobs: bool = True,
    top_logprobs: int = 5,
    timeout: int = 180,
    on_retry: Optional[Callable] = None,
) -> Dict:
    """One streamed completion, returning text plus full telemetry.

    Returns a dict with: text, ttft, e2e, prompt_tokens, completion_tokens,
    cost, provider, logprobs (list or None), logprobs_available, error.

    TTFT is measured to the first *content* chunk, not the first SSE frame:
    providers often emit a role-only opening delta, and counting that would
    understate TTFT by the width of one network hop.
    """
    params = params or {}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": params.get("temperature", 0.0),
        "top_p": params.get("top_p", 1.0),
        "max_tokens": params.get("max_tokens", 384),
        "stream": True,
        "usage": {"include": True},
        "provider": {"sort": OPENROUTER_PROVIDER_SORT, "allow_fallbacks": True},
    }
    for key in ("frequency_penalty", "presence_penalty", "seed", "top_k"):
        if params.get(key) is not None:
            payload[key] = params[key]
    if request_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs

    attempt = 0
    while True:
        result = _do_stream(model, payload, api_key, timeout)
        status = result.get("status_code")
        # `retries` feeds the Stage 9 reliability block: a run that silently
        # retried a third of its calls looks identical in the accuracy table.
        result["retries"] = attempt
        if status not in RETRYABLE or attempt >= OPENROUTER_MAX_RETRIES:
            return result
        attempt += 1
        wait = _sleep_for(attempt, None)
        if on_retry:
            on_retry(attempt, wait, status)
        time.sleep(wait)


def _do_stream(model: str, payload: Dict, api_key: str, timeout: int) -> Dict:
    out = {
        "text": "", "ttft": None, "e2e": None,
        "prompt_tokens": None, "completion_tokens": None, "cost": None,
        "reasoning_tokens": None, "cached_tokens": None,
        "provider": None, "logprobs": None, "logprobs_available": False,
        "error": None, "status_code": None, "chunks": 0, "retries": 0,
    }
    start = time.perf_counter()
    chunks: List[str] = []
    logprob_entries: List[Dict] = []

    try:
        resp = requests.post(
            OPENROUTER_BASE_URL, headers=_headers(api_key),
            json=payload, stream=True, timeout=timeout,
        )
        out["status_code"] = resp.status_code
        if resp.status_code != 200:
            out["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            out["e2e"] = time.perf_counter() - start
            return out

        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", "replace")
            if line.startswith(":"):          # SSE comment / keep-alive
                continue
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body.strip() == "[DONE]":
                break
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                continue

            if event.get("provider"):
                out["provider"] = event["provider"]
            usage = event.get("usage")
            if usage:
                out["prompt_tokens"] = usage.get("prompt_tokens")
                out["completion_tokens"] = usage.get("completion_tokens")
                out["cost"] = usage.get("cost")
                # Stage 7 detail. Present only when the provider accounts for
                # them; 0 is a real answer for a non-reasoning model.
                ctd = usage.get("completion_tokens_details") or {}
                if "reasoning_tokens" in ctd:
                    out["reasoning_tokens"] = ctd.get("reasoning_tokens")
                ptd = usage.get("prompt_tokens_details") or {}
                if "cached_tokens" in ptd:
                    out["cached_tokens"] = ptd.get("cached_tokens")

            for choice in event.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if out["ttft"] is None:
                        out["ttft"] = time.perf_counter() - start
                    chunks.append(piece)
                    out["chunks"] += 1
                lp = choice.get("logprobs")
                if lp and lp.get("content"):
                    logprob_entries.extend(lp["content"])

        out["e2e"] = time.perf_counter() - start
        out["text"] = "".join(chunks).strip()
        if logprob_entries:
            out["logprobs"] = logprob_entries
            out["logprobs_available"] = True
        if out["completion_tokens"] is None and out["chunks"]:
            # Fall back to chunk count so TPOT stays computable; marked
            # approximate by the absence of a provider-reported count.
            out["completion_tokens"] = out["chunks"]
    except Exception as exc:
        out["e2e"] = time.perf_counter() - start
        out["error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------

def probe_capabilities(models: List[str], api_key: str) -> Dict[str, Dict]:
    """Ask each model one trivial question to see what telemetry we can get.

    Run once at the start of an evaluation; the result belongs in the report
    so readers know why a calibration column is empty for some models.
    """
    caps = {}
    for model in models:
        res = stream_completion(
            model, "Reply with the single word: ok", api_key,
            params={"max_tokens": 5, "temperature": 0.0},
            request_logprobs=True, top_logprobs=5, timeout=60,
        )
        caps[model] = {
            "reachable": res["error"] is None,
            "provider": res.get("provider"),
            "logprobs": res["logprobs_available"],
            "streaming_ttft": res["ttft"] is not None,
            "usage_accounting": res["prompt_tokens"] is not None,
            "cost_accounting": res["cost"] is not None,
            "error": res["error"],
        }
    return caps


# ---------------------------------------------------------------------------
# Turning logprobs into the probabilities the calibration layer wants
# ---------------------------------------------------------------------------

def option_probs_from_logprobs(
    logprobs: Optional[List[Dict]],
    letters=("A", "B", "C", "D"),
) -> Optional[List[float]]:
    """Recover a distribution over A/B/C/D from the answer token's top-k.

    Scans for the first generated token whose top-k alternatives contain at
    least two option letters - that is the position where the model actually
    committed to an answer - then renormalises those letters to sum to 1.

    IMPORTANT - why `score_options()` exists alongside this
    -------------------------------------------------------
    Measured behaviour (OpenRouter, Sep 2026): when max_tokens > 1 the upstream
    provider returns logprobs for the FINAL token only, which is the EOS token,
    whose distribution says nothing about the answer. Only at max_tokens = 1
    does the returned logprob entry correspond to the answer letter itself.

    So this parser works on the output of `score_options()`, and returns None -
    not a uniform guess - on a normal chain-of-thought response.
    """
    if not logprobs:
        return None
    import math

    for entry in logprobs:
        tops = entry.get("top_logprobs") or []
        found = {}
        for cand in tops:
            tok = (cand.get("token") or "").strip().upper().lstrip("(*[ ")
            if tok[:1] in letters and tok[:1] not in found:
                found[tok[:1]] = cand.get("logprob")
        if len(found) >= 2:
            probs = []
            for letter in letters:
                lp = found.get(letter)
                probs.append(math.exp(lp) if lp is not None else 0.0)
            total = sum(probs)
            if total > 0:
                return [p / total for p in probs]
    return None


def score_options(
    model: str,
    prompt: str,
    api_key: str,
    api_letters=("A", "B", "C", "D"),
    top_logprobs: int = 10,
    timeout: int = 90,
) -> Dict:
    """Single-token constrained scoring - the calibration pass.

    Forces max_tokens=1 so the one generated token *is* the answer letter, then
    reads its top-k to recover P(A), P(B), P(C), P(D). This is the standard
    letter-scoring protocol (as in lm-evaluation-harness) and the only way to
    obtain calibration numbers from a chat endpoint that will not expose
    per-token logprobs for a longer generation.

    Deliberately a *separate, cheap pass* from the main reasoning evaluation:
      - the reasoning pass measures accuracy the way the model would really be
        used (chain of thought, then an answer);
      - this pass measures how much probability mass sat on the right letter.
    Merging them would force a choice between the two, and they answer
    different questions.

    Returns option_probs (or None), the argmax letter, confidence, and cost.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "provider": {"sort": OPENROUTER_PROVIDER_SORT, "allow_fallbacks": True},
    }
    out = {
        "option_probs": None, "predicted_letter": None, "confidence": None,
        "available": False, "cost": None, "provider": None, "error": None,
    }
    try:
        resp = requests.post(
            OPENROUTER_BASE_URL, headers=_headers(api_key),
            json=payload, timeout=timeout,
        )
        if resp.status_code != 200:
            out["error"] = f"HTTP {resp.status_code}: {resp.text[:160]}"
            return out
        data = resp.json()
        out["provider"] = data.get("provider")
        out["cost"] = (data.get("usage") or {}).get("cost")
        choice = (data.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        out["predicted_letter"] = text[:1].upper() if text else None

        logprobs = (choice.get("logprobs") or {}).get("content")
        probs = option_probs_from_logprobs(logprobs, api_letters)
        if probs:
            out["option_probs"] = probs
            out["confidence"] = max(probs)
            out["available"] = True
            out["predicted_letter"] = api_letters[probs.index(max(probs))]
    except Exception as exc:
        out["error"] = str(exc)
    return out


def score_next_token(
    model: str,
    prompt: str,
    api_key: str,
    target_word: str,
    top_logprobs: int = 20,
    timeout: int = 90,
) -> Dict:
    """Single-token scoring for LAMBADA: P(target) and the target's rank.

    Same constraint as `score_options`: only at max_tokens=1 does the returned
    logprob entry describe the token we care about. Verified working - for
    "she picked up the phone and dialed his" the target "number" comes back at
    rank 1 with logprob -0.01.

    Matching is on the target's first sub-token, since a multi-token target can
    never appear whole in one top-k list. That makes this an answer to "was the
    right continuation being considered?" rather than "would the whole word
    have been produced?" - the distinction is recorded in `match` below.
    """
    import math

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "provider": {"sort": OPENROUTER_PROVIDER_SORT, "allow_fallbacks": True},
    }
    out = {
        "available": False, "target_prob": None, "target_rank": None,
        "topk_probs": None, "match": None, "cost": None, "error": None,
    }
    try:
        resp = requests.post(
            OPENROUTER_BASE_URL, headers=_headers(api_key),
            json=payload, timeout=timeout,
        )
        if resp.status_code != 200:
            out["error"] = f"HTTP {resp.status_code}"
            return out
        data = resp.json()
        out["cost"] = (data.get("usage") or {}).get("cost")
        entries = ((data.get("choices") or [{}])[0].get("logprobs") or {}).get("content")
        if not entries:
            return out

        tops = entries[0].get("top_logprobs") or []
        if not tops:
            return out

        needle = (target_word or "").strip().lower()
        probs = []
        for i, cand in enumerate(tops, start=1):
            tok = (cand.get("token") or "").strip().lower()
            lp = cand.get("logprob")
            p = math.exp(lp) if lp is not None else 0.0
            probs.append(p)
            if out["target_rank"] is None and tok and needle:
                if tok == needle:
                    out["target_rank"], out["target_prob"], out["match"] = i, p, "exact"
                elif needle.startswith(tok):
                    out["target_rank"], out["target_prob"], out["match"] = i, p, "prefix"

        out["topk_probs"] = probs
        out["available"] = True
    except Exception as exc:
        out["error"] = str(exc)
    return out


def target_rank_from_logprobs(
    logprobs: Optional[List[Dict]],
    target_word: str,
) -> Optional[int]:
    """Rank of a LAMBADA target inside the first generated token's top-k.

    Compares on the first sub-token of the target, since a multi-token target
    can never appear whole in a single top-k list. Documented as an
    approximation: it answers "was the right continuation being considered?",
    not "would the whole word have been produced?".
    """
    if not logprobs or not target_word:
        return None
    needle = target_word.strip().lower()
    tops = (logprobs[0].get("top_logprobs") or [])
    for i, cand in enumerate(tops, start=1):
        tok = (cand.get("token") or "").strip().lower()
        if tok and (tok == needle or needle.startswith(tok)):
            return i
    return None


def sequence_logprob(logprobs: Optional[List[Dict]]) -> Optional[float]:
    """Sum of token logprobs - the log-likelihood of what was generated."""
    if not logprobs:
        return None
    vals = [e.get("logprob") for e in logprobs if e.get("logprob") is not None]
    return sum(vals) if vals else None


def sequence_perplexity(logprobs: Optional[List[Dict]]) -> Optional[float]:
    """Perplexity of the generated sequence."""
    import math
    if not logprobs:
        return None
    vals = [e.get("logprob") for e in logprobs if e.get("logprob") is not None]
    if not vals:
        return None
    mean_nll = -sum(vals) / len(vals)
    return math.exp(mean_nll) if mean_nll < 700 else float("inf")
