import random
import time

import requests

from config import (
    OPENROUTER_MAX_RETRIES,
    OPENROUTER_RETRY_BASE_DELAY,
    OPENROUTER_RETRY_MAX_DELAY,
)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def post_with_retry(url, headers, json_payload, timeout, on_retry=None):
    """POST to OpenRouter, retrying on 429 (rate limit) and 5xx with backoff.

    Honors the response's `Retry-After` header when present (OpenRouter sends
    this on 429s); otherwise backs off exponentially with jitter. Returns the
    final response either way, so callers keep using `resp.raise_for_status()`
    as before - only the outcome of the last attempt is surfaced as an error.

    `on_retry(attempt, wait_seconds, status_code)`, when given, is called
    before each retry sleep so callers can surface it in a live log.
    """
    attempt = 0
    while True:
        resp = requests.post(url, headers=headers, json=json_payload, timeout=timeout)
        if resp.status_code not in RETRYABLE_STATUS_CODES:
            return resp

        attempt += 1
        if attempt > OPENROUTER_MAX_RETRIES:
            return resp

        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            try:
                wait = float(retry_after)
            except ValueError:
                wait = OPENROUTER_RETRY_BASE_DELAY * (2 ** (attempt - 1))
        else:
            wait = min(
                OPENROUTER_RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                OPENROUTER_RETRY_MAX_DELAY,
            )
        wait += random.uniform(0, 0.5)

        if on_retry:
            on_retry(attempt, wait, resp.status_code)
        time.sleep(wait)
