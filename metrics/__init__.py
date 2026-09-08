"""Metric layer for the SLM benchmark.

Organised by the taxonomy the report uses, one module per family:

    quality       accuracy family, macro/micro, error and parse-failure rates
    calibration   correct-option probability, NLL, PPL, entropy, ECE, Brier, MRR
    consistency   answer stability, self-consistency, seed stability
    robustness    seeded input perturbations and the resulting accuracy deltas
    context       context ablation / utilisation / position sensitivity (LAMBADA)
    tokenization  tokens per target word, fragmentation, subword match (LAMBADA)
    systems       TTFT / TPOT / E2E, throughput, cost, and analytic memory+energy
    aggregate     composes the above and ranks models

Configuration variables (TP/PP/DP/SP/CP/EP, dtype, batch size, ...) live in
`benchmark_config.py`, deliberately outside this package: they are inputs to a
run, not measurements of a model.
"""

from . import (
    aggregate,
    calibration,
    consistency,
    context,
    quality,
    robustness,
    systems,
    tokenization,
)

__all__ = [
    "aggregate", "calibration", "consistency", "context",
    "quality", "robustness", "systems", "tokenization",
]

METRIC_FAMILIES = {
    "task_quality": "Was the answer right?",
    "probability": "How confident was it, and was that confidence earned?",
    "consistency": "Would it answer the same way again?",
    "robustness": "Does the score survive a harmless change to the input?",
    "context": "Is it actually reading the passage? (LAMBADA)",
    "tokenization": "Is the tokenizer handicapping it? (LAMBADA)",
    "systems": "What did it cost in time, memory and money?",
}
