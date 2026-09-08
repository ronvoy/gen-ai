"""Systems metrics: latency, throughput, cost, memory, energy.

MEASURABILITY IS THE POINT OF THIS MODULE.

This benchmark talks to models over OpenRouter, i.e. someone else's GPUs. That
makes a hard split:

  Directly measured (client-side, real numbers)
      TTFT, TPOT, E2E latency, prefill/decode throughput, token counts,
      monetary cost - all observable from a streaming HTTP response.

  Structurally unobservable through a hosted API
      VRAM, KV-cache size, energy, communication overhead, and the effect of
      TP/PP/DP/SP/CP/EP. We are one tenant on a shared, auto-scaled backend
      whose batch composition and parallelism layout we neither set nor see.

Every field carries a `source` tag so a reader can never mistake one for the
other. Where a value is unobservable we emit `None` plus a reason - an
analytic estimate is offered only when the caller supplies the hardware facts
it needs, and it is then labelled `analytic_model`, never `measured`.

Fabricating a VRAM or joules figure for a hosted endpoint would be the single
easiest way to make this whole report untrustworthy.
"""

import math
import time
from typing import Dict, List, Optional, Sequence

# Provenance tags used throughout.
SOURCE_MEASURED = "measured"           # observed by this client
SOURCE_PROVIDER = "provider_reported"  # taken from the API usage payload
SOURCE_ANALYTIC = "analytic_model"     # computed from user-supplied hardware
SOURCE_UNAVAILABLE = "unavailable"     # cannot be known in this deployment


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Linear-interpolated percentile. p in 0..100."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return vals[int(k)]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def latency_stats(samples: Sequence[float], label: str) -> Optional[Dict]:
    """Mean plus the tail percentiles that actually matter for serving.

    p95/p99 are reported because a mean hides the tail, and for interactive
    serving the tail is the user experience.
    """
    vals = [v for v in samples if v is not None]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 0.0
    return {
        "metric": label,
        "mean": round(mean, 4),
        "std": round(math.sqrt(var), 4),
        "min": round(min(vals), 4),
        "p50": round(percentile(vals, 50), 4),
        "p95": round(percentile(vals, 95), 4),
        "p99": round(percentile(vals, 99), 4),
        "max": round(max(vals), 4),
        "n": len(vals),
        "source": SOURCE_MEASURED,
    }


def compute_tpot(e2e: float, ttft: float, completion_tokens: int) -> Optional[float]:
    """Time Per Output Token, in seconds.

    Excludes the first token: TTFT covers prefill, TPOT should describe the
    steady-state decode rate only. Needs >= 2 tokens to be meaningful.
    """
    if completion_tokens is None or completion_tokens < 2:
        return None
    if e2e is None or ttft is None or e2e <= ttft:
        return None
    return (e2e - ttft) / (completion_tokens - 1)


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------

def prefill_throughput(prompt_tokens: int, ttft: float) -> Optional[float]:
    """Prompt tokens per second during prefill.

    TTFT bundles network round-trip and queueing with actual prefill, so on a
    hosted endpoint this is an *upper bound on latency*, i.e. a lower bound on
    true prefill speed. Documented rather than silently reported as exact.
    """
    if not prompt_tokens or not ttft or ttft <= 0:
        return None
    return prompt_tokens / ttft


def decode_throughput(completion_tokens: int, decode_seconds: float) -> Optional[float]:
    """Generated tokens per second during decode."""
    if not completion_tokens or not decode_seconds or decode_seconds <= 0:
        return None
    return completion_tokens / decode_seconds


def request_throughput(n_requests: int, wall_seconds: float) -> Optional[float]:
    """Completed requests per second across the whole run (system-level)."""
    if not n_requests or not wall_seconds or wall_seconds <= 0:
        return None
    return n_requests / wall_seconds


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def cost_stats(costs: Sequence[float], correct_count: int = 0) -> Optional[Dict]:
    """Aggregate real spend, reported by the API per request.

    `cost_per_correct_answer` is the number that actually drives model choice
    in production: a cheaper model that is wrong twice as often is not cheaper.
    """
    vals = [c for c in costs if c is not None]
    if not vals:
        return None
    total = sum(vals)
    return {
        "total_usd": round(total, 6),
        "mean_per_request_usd": round(total / len(vals), 8),
        "per_1k_requests_usd": round(total / len(vals) * 1000, 4),
        "cost_per_correct_answer_usd": round(
            total / correct_count, 6
        ) if correct_count else None,
        "n_requests": len(vals),
        "source": SOURCE_PROVIDER,
    }


# ---------------------------------------------------------------------------
# Memory - analytic only, never measured through a hosted API
# ---------------------------------------------------------------------------

def kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """Analytic KV-cache size in bytes.

        2 (K and V) x layers x kv_heads x head_dim x seq_len x batch x dtype

    Exact for a standard decoder given the architecture numbers - but those
    numbers describe the *model*, not the deployment we measured. Use only
    for capacity planning, and only with the analytic label attached.
    """
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes


def model_weight_bytes(n_params: float, dtype_bytes: int = 2) -> int:
    """Parameter memory. Activations and fragmentation are extra."""
    return int(n_params * dtype_bytes)


def estimate_memory(
    arch: Optional[Dict],
    seq_len: int = 2048,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> Dict:
    """VRAM estimate, or an explicit refusal when we lack the facts.

    `arch` must carry n_params, n_layers, n_kv_heads, head_dim. Without it we
    return `available: False` rather than inventing a plausible number.
    """
    required = ("n_params", "n_layers", "n_kv_heads", "head_dim")
    if not arch or not all(k in arch for k in required):
        return {
            "available": False,
            "source": SOURCE_UNAVAILABLE,
            "reason": (
                "VRAM/KV-cache cannot be observed through a hosted inference API; "
                "supply model architecture in MODEL_ARCH to compute an analytic estimate"
            ),
        }

    weights = model_weight_bytes(arch["n_params"], dtype_bytes)
    kv = kv_cache_bytes(
        arch["n_layers"], arch["n_kv_heads"], arch["head_dim"],
        seq_len, batch_size, dtype_bytes,
    )
    gb = 1024 ** 3
    return {
        "available": True,
        "source": SOURCE_ANALYTIC,
        "assumptions": {
            "seq_len": seq_len, "batch_size": batch_size,
            "dtype_bytes": dtype_bytes,
        },
        "weights_gb": round(weights / gb, 3),
        "kv_cache_gb": round(kv / gb, 4),
        "kv_cache_mb_per_1k_tokens": round(
            kv_cache_bytes(arch["n_layers"], arch["n_kv_heads"], arch["head_dim"],
                           1000, 1, dtype_bytes) / (1024 ** 2), 3
        ),
        "estimated_total_gb": round((weights + kv) * 1.15 / gb, 3),
        "note": "total includes a flat 15% allowance for activations and allocator overhead",
    }


# ---------------------------------------------------------------------------
# Energy - analytic only
# ---------------------------------------------------------------------------

def estimate_energy(
    total_seconds: float,
    device_tdp_watts: Optional[float] = None,
    utilisation: float = 0.7,
) -> Dict:
    """Energy estimate from wall time and a device TDP the caller supplies.

    Genuinely measuring this needs NVML/RAPL on the serving host, which we do
    not have. With no TDP given we say so rather than guessing.
    """
    if not device_tdp_watts:
        return {
            "available": False,
            "source": SOURCE_UNAVAILABLE,
            "reason": (
                "energy requires on-host power telemetry (NVML/RAPL); "
                "not observable for a remote hosted endpoint. "
                "Set device_tdp_watts for an analytic estimate."
            ),
        }
    joules = device_tdp_watts * utilisation * total_seconds
    return {
        "available": True,
        "source": SOURCE_ANALYTIC,
        "assumptions": {
            "device_tdp_watts": device_tdp_watts,
            "assumed_utilisation": utilisation,
        },
        "joules": round(joules, 2),
        "wh": round(joules / 3600.0, 4),
    }


def energy_per_token(energy_wh: Optional[float], tokens: int) -> Optional[float]:
    """Watt-hours per generated token - the efficiency figure worth citing."""
    if energy_wh is None or not tokens:
        return None
    return energy_wh / tokens


# ---------------------------------------------------------------------------
# Block builder
# ---------------------------------------------------------------------------

def build_api_performance_block(
    records: Sequence[Dict],
    wall_seconds: Optional[float] = None,
) -> Dict:
    """Stage 6 - API Performance: latency and throughput of the service.

    Every figure is client-measured from the streamed response.
    """
    ttfts = [r.get("ttft") for r in records if r.get("ttft") is not None]
    e2es = [r.get("e2e", r.get("time")) for r in records]
    e2es = [v for v in e2es if v is not None]

    tpots, prefills, decodes = [], [], []
    for r in records:
        ttft, ct, pt = r.get("ttft"), r.get("completion_tokens"), r.get("prompt_tokens")
        e2e = r.get("e2e", r.get("time"))
        tp = compute_tpot(e2e, ttft, ct) if (e2e and ttft and ct) else None
        if tp:
            tpots.append(tp)
        pf = prefill_throughput(pt, ttft) if (pt and ttft) else None
        if pf:
            prefills.append(pf)
        if e2e and ttft and ct and e2e > ttft:
            dt = decode_throughput(ct, e2e - ttft)
            if dt:
                decodes.append(dt)

    total_tokens = sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
                       for r in records)
    return {
        "available": bool(e2es),
        "latency": {
            "ttft": latency_stats(ttfts, "TTFT (s)"),
            "e2e": latency_stats(e2es, "End-to-end (s)"),
            "tpot": latency_stats(tpots, "TPOT (s/token)"),
        },
        "throughput": {
            "prefill_tokens_per_s": _r(_mean(prefills), 2),
            "decode_tokens_per_s": _r(_mean(decodes), 2),
            "total_tokens_per_s": _r(
                total_tokens / wall_seconds, 2) if wall_seconds else None,
            "requests_per_s": _r(
                request_throughput(len(records), wall_seconds), 4) if wall_seconds else None,
            "items_per_s": _r(
                request_throughput(len(records), wall_seconds), 4) if wall_seconds else None,
        },
        "wall_seconds": _r(wall_seconds, 2) if wall_seconds else None,
        "source": SOURCE_MEASURED,
        "caveat": (
            "TTFT includes network round-trip and provider queueing, so "
            "TTFT-derived prefill throughput is a lower bound on true prefill speed"
        ),
    }


def build_token_efficiency_block(
    records: Sequence[Dict],
    correct_count: int = 0,
) -> Dict:
    """Stage 7 - Token Efficiency: what the run consumed.

    Counts are provider-reported (the API `usage` payload), including reasoning
    and cached-prompt tokens where the provider accounts for them.
    """
    n = len(records)
    if not n:
        return {"available": False, "reason": "no requests recorded"}

    prompt = sum(r.get("prompt_tokens") or 0 for r in records)
    completion = sum(r.get("completion_tokens") or 0 for r in records)
    reasoning = sum(r.get("reasoning_tokens") or 0 for r in records)
    cached = sum(r.get("cached_tokens") or 0 for r in records)
    total = prompt + completion

    block = {
        "available": True,
        "prompt_tokens_total": prompt,
        "completion_tokens_total": completion,
        "total_tokens": total,
        "mean_prompt_tokens": _r(prompt / n, 1),
        "mean_completion_tokens": _r(completion / n, 1),
        "tokens_per_item": _r(total / n, 1),
        "tokens_per_correct_answer": _r(total / correct_count, 1) if correct_count else None,
        "source": SOURCE_PROVIDER,
    }
    # Reported whenever the provider accounts for them; 0 is a real answer for
    # a non-reasoning model, so it is shown rather than hidden.
    if any(r.get("reasoning_tokens") is not None for r in records):
        block["reasoning_tokens_total"] = reasoning
        block["mean_reasoning_tokens"] = _r(reasoning / n, 1)
        block["reasoning_token_share"] = _r(
            reasoning / completion, 4) if completion else None
    if any(r.get("cached_tokens") is not None for r in records):
        block["cached_prompt_tokens_total"] = cached
        block["prompt_cache_hit_rate"] = _r(cached / prompt, 4) if prompt else None
    return block


def build_economics_block(
    records: Sequence[Dict],
    correct_count: int = 0,
) -> Dict:
    """Stage 8 - Economics: real money, as reported by the API."""
    costs = [r.get("cost") for r in records if r.get("cost") is not None]
    if not costs:
        return {
            "available": False,
            "reason": "provider did not report per-request cost for this run",
        }

    total_cost = sum(costs)
    tokens = sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
                 for r in records)
    stats = cost_stats(costs, correct_count) or {}
    block = {
        "available": True,
        "total_usd": stats.get("total_usd"),
        "mean_per_request_usd": stats.get("mean_per_request_usd"),
        "per_1k_requests_usd": stats.get("per_1k_requests_usd"),
        "cost_per_correct_answer_usd": stats.get("cost_per_correct_answer_usd"),
        "n_requests": len(costs),
        "source": SOURCE_PROVIDER,
    }
    if tokens:
        block["cost_per_1k_tokens_usd"] = _r(total_cost / tokens * 1_000, 6)
        block["cost_per_1m_tokens_usd"] = _r(total_cost / tokens * 1_000_000, 4)
    if correct_count and total_cost:
        # Accuracy bought per dollar - the headline efficiency ratio.
        block["correct_answers_per_usd"] = _r(correct_count / total_cost, 1)
    return block


# Retained for the self-hosted path only. build_systems_block is deliberately
# NOT called by the OpenRouter runner: memory and energy cannot be measured
# through a hosted API, and stage 10 is excluded from this benchmark entirely
# (see metrics/taxonomy.py EXCLUDED_STAGE).
def build_selfhosted_systems_block(
    records: Sequence[Dict],
    wall_seconds: Optional[float] = None,
    arch: Optional[Dict] = None,
    device_tdp_watts: Optional[float] = None,
) -> Dict:
    """Analytic memory/energy estimates for a self-hosted deployment."""
    completion_tokens = sum(r.get("completion_tokens") or 0 for r in records)
    block = {
        "memory": estimate_memory(arch),
        "energy": estimate_energy(wall_seconds or 0, device_tdp_watts),
        "deployment": "self_hosted",
    }
    if completion_tokens and block["energy"].get("available"):
        block["energy"]["wh_per_1k_output_tokens"] = _r(
            energy_per_token(block["energy"]["wh"], completion_tokens) * 1000, 6)
    return block


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _r(value: Optional[float], places: int = 4) -> Optional[float]:
    return None if value is None else round(value, places)
