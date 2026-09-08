"""Stage 9 - Reliability: did the calls actually succeed?

Every metric here is observable client-side, needs no extra API calls, and is
otherwise invisible: a run that quietly retried a third of its requests and
dropped four to timeouts produces the same accuracy table as a clean one. The
difference matters for anyone deciding whether to depend on a provider.

Kept separate from Task Quality on purpose. "The model answered wrongly" and
"the call never came back" are different failures with different owners - one
belongs to the model, the other to the serving path.
"""

import math
from typing import Dict, List, Optional, Sequence


def _rate(count: int, total: int) -> Optional[float]:
    return (count / total) if total else None


def classify_error(error: Optional[str]) -> Optional[str]:
    """Bucket a raw error string into a reportable failure class."""
    if not error:
        return None
    text = str(error).lower()
    if "429" in text or "rate-limit" in text or "rate limit" in text:
        return "rate_limited"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if any(code in text for code in ("500", "502", "503", "504")):
        return "server_error"
    if "connection" in text or "resolve" in text or "ssl" in text:
        return "network"
    if "401" in text or "403" in text:
        return "auth"
    if "400" in text or "422" in text:
        return "bad_request"
    return "other"


def build_reliability_block(
    records: Sequence[Dict],
    answer_key: str = "predicted_letter",
    retry_events: Optional[Sequence[Dict]] = None,
) -> Dict:
    """Assemble the Reliability block from per-request records.

    Records may carry `error` (string or None), `retries` (int) and `provider`.
    `invalid output` is counted separately from `error`: a 200 response whose
    body could not be parsed into an answer is a model/prompt failure, not a
    transport one, and conflating them hides which to fix.
    """
    total = len(records)
    if not total:
        return {"available": False, "reason": "no requests recorded"}

    errors = [r.get("error") for r in records]
    failed = [e for e in errors if e]
    classes: Dict[str, int] = {}
    for e in failed:
        cls = classify_error(e) or "other"
        classes[cls] = classes.get(cls, 0) + 1

    # A response that arrived but carried nothing usable.
    invalid = sum(
        1 for r in records
        if not r.get("error") and not r.get(answer_key)
    )

    retries = sum(int(r.get("retries") or 0) for r in records)
    retried_requests = sum(1 for r in records if (r.get("retries") or 0) > 0)

    providers = [r.get("provider") for r in records if r.get("provider")]
    provider_counts: Dict[str, int] = {}
    for p in providers:
        provider_counts[p] = provider_counts.get(p, 0) + 1

    successes = total - len(failed)

    block = {
        "available": True,
        "n_requests": total,
        "successful_requests": successes,
        "success_rate": round(_rate(successes, total) or 0, 4),
        "failure_rate": round(_rate(len(failed), total) or 0, 4),
        "invalid_output_rate": round(_rate(invalid, total) or 0, 4),
        "timeout_rate": round(_rate(classes.get("timeout", 0), total) or 0, 4),
        "rate_limit_rate": round(_rate(classes.get("rate_limited", 0), total) or 0, 4),
        "server_error_rate": round(_rate(classes.get("server_error", 0), total) or 0, 4),
        "error_classes": classes,
        "retry_total": retries,
        "retried_request_rate": round(_rate(retried_requests, total) or 0, 4),
        "retries_per_request": round(retries / total, 4),
        "providers_used": provider_counts,
        # More than one provider in a single run means OpenRouter failed over
        # mid-run: latency and even quality figures then mix two backends.
        "provider_failover": len(provider_counts) > 1,
        "source": "measured",
    }

    if block["provider_failover"]:
        block["failover_warning"] = (
            "requests in this run were served by more than one provider; "
            "latency and probability coverage may mix backends"
        )
    return block


def reproducibility_score(
    seed_accuracies: Sequence[float],
    repeat_stability: Optional[float] = None,
) -> Optional[Dict]:
    """A single 0-1 reproducibility figure with its inputs exposed.

    Combines across-seed variance with within-prompt answer stability. Scored
    so 1.0 means "same input, same output, every time"; the components are
    reported alongside so a low score can be attributed.
    """
    parts = {}
    if seed_accuracies and len(seed_accuracies) >= 2:
        mean = sum(seed_accuracies) / len(seed_accuracies)
        var = sum((a - mean) ** 2 for a in seed_accuracies) / (len(seed_accuracies) - 1)
        std = math.sqrt(var)
        # 10pp of spread is treated as fully irreproducible.
        parts["seed_component"] = max(0.0, 1.0 - min(std / 0.10, 1.0))
        parts["seed_std"] = round(std, 4)
    if repeat_stability is not None:
        parts["stability_component"] = repeat_stability

    components = [v for k, v in parts.items() if k.endswith("_component")]
    if not components:
        return None
    return {
        "reproducibility_score": round(sum(components) / len(components), 4),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in parts.items()},
        "interpretation": (
            "1.0 = identical output for identical input. Anything materially "
            "below that at temperature 0 comes from the serving stack "
            "(batching non-determinism, provider failover), not from sampling."
        ),
    }
