"""OpenRouter HTTP client with adaptive, *shared* rate-limit handling.

Why this is not just a retry loop
---------------------------------
A per-request retry loop treats every 429 as news. During a sustained upstream
rate limit that is the worst possible behaviour: request N burns its whole
backoff ladder (~92s) discovering the provider is throttling, gives up, and
request N+1 immediately starts over and discovers the same thing. A 2,850-item
run spends hours in `time.sleep` and still records most items as failures.

So the throttle state is module-level and shared:

  * `cooldown_until` - when any request sees a 429, every other request waits
    for the same window instead of re-discovering it.
  * `min_interval`   - adaptive spacing between requests. Grows multiplicatively
    on a 429, decays on sustained success, so the client settles just under the
    rate the provider will actually serve.
  * provider re-routing - `provider.sort` is dropped after repeated 429s so
    OpenRouter can hand the request to a different backend. Sorting by
    throughput otherwise keeps steering into the same congested provider,
    which is what produces a sustained 429 streak in the first place.

`post_with_retry` keeps its original signature, so existing callers are
unaffected.
"""

import random
import threading
import time

import requests

from config import (
    OPENROUTER_MAX_RETRIES,
    OPENROUTER_RETRY_BASE_DELAY,
    OPENROUTER_RETRY_MAX_DELAY,
)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Pacing bounds. The floor keeps a healthy run from being slowed for no reason;
# the ceiling stops a long 429 streak from stretching one request into minutes.
MIN_INTERVAL_FLOOR = 0.0
MIN_INTERVAL_CEILING = 8.0
INTERVAL_GROWTH = 1.6          # multiply spacing on each 429
INTERVAL_DECAY = 0.9           # shrink it after a clean response
DECAY_AFTER_SUCCESSES = 5      # only decay once the provider looks healthy
MAX_COOLDOWN = 60.0            # cap on a single shared pause


class _AdaptiveThrottle:
    """Process-wide pacing state shared by every request."""

    def __init__(self):
        self._lock = threading.Lock()
        self.min_interval = MIN_INTERVAL_FLOOR
        self.cooldown_until = 0.0
        self.last_request = 0.0
        self.consecutive_ok = 0
        self.total_429 = 0
        self.total_waited = 0.0

    def before_request(self):
        """Block until it is this request's turn. Returns seconds slept."""
        while True:
            with self._lock:
                now = time.monotonic()
                wait_cooldown = max(0.0, self.cooldown_until - now)
                wait_spacing = max(
                    0.0, (self.last_request + self.min_interval) - now
                )
                wait = max(wait_cooldown, wait_spacing)
                if wait <= 0:
                    self.last_request = now
                    return 0.0
            # Sleep outside the lock so other threads can make progress.
            time.sleep(min(wait, 5.0))
            with self._lock:
                self.total_waited += min(wait, 5.0)

    def on_rate_limited(self, retry_after=None):
        """Record a 429 and widen the shared pause."""
        with self._lock:
            self.total_429 += 1
            self.consecutive_ok = 0
            self.min_interval = min(
                MIN_INTERVAL_CEILING,
                max(0.5, self.min_interval * INTERVAL_GROWTH),
            )
            pause = retry_after if retry_after else self.min_interval * 2
            pause = min(pause, MAX_COOLDOWN)
            self.cooldown_until = max(
                self.cooldown_until, time.monotonic() + pause
            )
            return pause, self.min_interval

    def on_success(self):
        with self._lock:
            self.consecutive_ok += 1
            if (self.consecutive_ok >= DECAY_AFTER_SUCCESSES
                    and self.min_interval > MIN_INTERVAL_FLOOR):
                self.min_interval = max(
                    MIN_INTERVAL_FLOOR, self.min_interval * INTERVAL_DECAY
                )
                self.consecutive_ok = 0

    def stats(self):
        with self._lock:
            return {
                "rate_limit_hits": self.total_429,
                "current_min_interval_s": round(self.min_interval, 2),
                "total_throttle_wait_s": round(self.total_waited, 1),
            }

    def reset(self):
        with self._lock:
            self.__init__()


THROTTLE = _AdaptiveThrottle()


def _retry_after_seconds(resp):
    """Seconds to wait, from whichever header the provider actually sent."""
    if resp is None:
        return None
    headers = resp.headers or {}

    raw = headers.get("Retry-After")
    if raw:
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass

    # OpenRouter forwards upstream reset hints under several spellings; the
    # value is sometimes an epoch (seconds or milliseconds), sometimes a delta.
    for key in ("X-RateLimit-Reset", "x-ratelimit-reset",
                "X-RateLimit-Reset-Requests"):
        raw = headers.get(key)
        if not raw:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 1e12:                 # epoch milliseconds
            return max(0.0, value / 1000.0 - time.time())
        if value > 1e9:                  # epoch seconds
            return max(0.0, value - time.time())
        return max(0.0, value)           # already a delta
    return None


def preflight(model, api_key, probes=4, log=print):
    """Sample a model's current 429 rate before committing to a long run.

    A 2,850-item benchmark against a congested provider takes hours and yields
    mostly failures. Four cheap probes up front turn that into a decision the
    caller can make in seconds. Returns a dict; `healthy` is False when the
    provider refused most probes.
    """
    from config import OPENROUTER_BASE_URL

    hdrs = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single letter A"}],
        "max_tokens": 3,
        "temperature": 0,
    }
    ok = refused = 0
    providers = set()
    for _ in range(probes):
        try:
            resp = requests.post(OPENROUTER_BASE_URL, headers=hdrs, json=body, timeout=45)
        except requests.RequestException:
            refused += 1
            continue
        if resp.status_code == 429:
            refused += 1
        elif resp.status_code == 200:
            ok += 1
            prov = resp.json().get("provider")
            if prov:
                providers.add(prov)
        time.sleep(0.3)

    rate = refused / probes if probes else 0.0
    result = {
        "model": model,
        "probes": probes,
        "ok": ok,
        "refused": refused,
        "refusal_rate": round(rate, 2),
        "providers": sorted(providers),
        "healthy": rate <= 0.25,
    }
    if log:
        if result["healthy"]:
            log(f"  {model}: OK ({ok}/{probes})"
                + (f" via {', '.join(result['providers'])}" if providers else ""))
        else:
            log(f"  {model}: THROTTLED - {refused}/{probes} refused with 429. "
                f"The run will still complete (requests are retried, not dropped) "
                f"but will be slow. Consider running this model later.")
    return result


def post_with_retry(url, headers, json_payload, timeout, on_retry=None):
    """POST to OpenRouter, coordinating backoff across all callers.

    Honours Retry-After when present, otherwise backs off exponentially with
    jitter, and shares the resulting pause with every other request through
    the module-level throttle. Returns the final response either way, so
    callers keep using `resp.raise_for_status()`.

    `on_retry(attempt, wait_seconds, status_code)` is called before each sleep.
    """
    payload = json_payload
    attempt = 0

    while True:
        THROTTLE.before_request()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException:
            # Network-level failure: treat like a retryable server error rather
            # than letting the exception escape mid-run.
            attempt += 1
            if attempt > OPENROUTER_MAX_RETRIES:
                raise
            wait = min(
                OPENROUTER_RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                OPENROUTER_RETRY_MAX_DELAY,
            ) + random.uniform(0, 0.5)
            if on_retry:
                on_retry(attempt, wait, "network")
            time.sleep(wait)
            continue

        if resp.status_code not in RETRYABLE_STATUS_CODES:
            THROTTLE.on_success()
            return resp

        attempt += 1
        if attempt > OPENROUTER_MAX_RETRIES:
            return resp

        if resp.status_code == 429:
            retry_after = _retry_after_seconds(resp)
            wait, interval = THROTTLE.on_rate_limited(retry_after)
            # After two refusals, stop steering to the same busy provider and
            # let OpenRouter pick any backend that can serve us.
            if attempt >= 2 and isinstance(payload, dict):
                provider = payload.get("provider")
                if isinstance(provider, dict) and "sort" in provider:
                    payload = dict(payload)
                    payload["provider"] = {
                        k: v for k, v in provider.items() if k != "sort"
                    }
                    payload["provider"]["allow_fallbacks"] = True
            if on_retry:
                on_retry(attempt, wait, resp.status_code)
        else:
            wait = min(
                OPENROUTER_RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                OPENROUTER_RETRY_MAX_DELAY,
            ) + random.uniform(0, 0.5)
            if on_retry:
                on_retry(attempt, wait, resp.status_code)
            time.sleep(wait)
