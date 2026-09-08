"""The benchmark's metric taxonomy: stages, priorities and observability.

Single source of truth for "what does this benchmark measure, and can this
deployment actually measure it?". The UI, the report and the runner all read
this rather than each keeping their own list.

Structure
---------
Stages 1-9 are the OpenRouter-observable benchmark. Stage 10 (hardware and
distributed) is deliberately NOT part of it: nothing in it can be observed
through a hosted inference API, so it lives here only as a documented
exclusion pointing at the self-hosted path.

Observability values
--------------------
    yes         measured directly from the API response
    conditional depends on the routed provider (log-probabilities, mainly)
    no          structurally impossible through a hosted API
"""

from typing import Dict, List, Optional

YES = "yes"
CONDITIONAL = "conditional"
NO = "no"

# Priority bands. P0 is what every run produces; P1 needs an extra pass or
# repeat; P2 is analysis that only becomes meaningful at larger sample sizes.
P0, P1, P2 = "P0", "P1", "P2"


STAGES: List[Dict] = [
    {
        "stage": 1,
        "key": "task_quality",
        "layer": "Task Quality",
        "purpose": "Measure model capability",
        "mmlu": YES, "lambada": YES, "openrouter": YES,
        "priority": P0,
        "subsections": [
            {"name": "Core Task Performance",
             "metrics": ["Accuracy", "Exact Match", "Error Rate",
                         "Normalised Accuracy", "Parse-Failure Rate"],
             "observable": YES},
            {"name": "MMLU Domain Performance",
             "metrics": ["Macro Accuracy", "Subject Accuracy",
                         "Category Accuracy", "Wilson CI"],
             "observable": YES, "benchmarks": ["mmlu"]},
            {"name": "LAMBADA Word Prediction",
             "metrics": ["Last-Word Accuracy", "Exact Target Match",
                         "Stem Match"],
             "observable": YES, "benchmarks": ["lambada"]},
            {"name": "Language Modeling",
             # Moved out of the unconditional band: perplexity and NLL are
             # derived from token log-probabilities, so they inherit the same
             # provider dependency as stage 2 and cannot be promised per run.
             "metrics": ["Perplexity", "NLL", "Cross Entropy"],
             "observable": CONDITIONAL,
             "note": "derived from logprobs - same provider dependency as stage 2"},
        ],
    },
    {
        "stage": 2,
        "key": "probability",
        "layer": "Probabilistic Quality",
        "purpose": "Measure confidence and probability quality",
        "mmlu": YES, "lambada": YES, "openrouter": CONDITIONAL,
        "priority": P1,
        "requires": "provider that returns token log-probabilities",
        "subsections": [
            {"name": "Probability",
             "metrics": ["Correct-Option Probability", "Target Probability",
                         "Log Probability", "Target Rank", "Probability Margin"],
             "observable": CONDITIONAL},
            {"name": "Uncertainty",
             "metrics": ["Entropy", "Normalised Entropy", "Prediction Entropy"],
             "observable": CONDITIONAL},
            {"name": "Calibration",
             "metrics": ["Confidence", "ECE", "MCE", "Brier Score",
                         "Confidence-Accuracy Correlation"],
             "observable": CONDITIONAL},
        ],
    },
    {
        "stage": 3,
        "key": "consistency",
        "layer": "Reasoning & Consistency",
        "purpose": "Measure stability and reasoning reliability",
        "mmlu": YES, "lambada": YES, "openrouter": YES,
        "priority": P1,
        "requires": "repeats > 1 and/or multiple seeds",
        "subsections": [
            {"name": "Answer Stability",
             "metrics": ["Prediction Stability", "Answer Agreement"], "observable": YES},
            {"name": "Self-Consistency",
             "metrics": ["Self-Consistency Accuracy", "Majority-Vote Accuracy"],
             "observable": YES},
            {"name": "Reproducibility",
             "metrics": ["Seed Stability", "Prediction Variance",
                         "Run-to-Run Variance"], "observable": YES},
        ],
    },
    {
        "stage": 4,
        "key": "context",
        "layer": "Context Behavior",
        "purpose": "Measure context usage and dependency",
        "mmlu": "experimental", "lambada": YES, "openrouter": YES,
        "priority": P1,
        "requires": "context ablation pass",
        "subsections": [
            {"name": "Context Utilization",
             "metrics": ["Context Utilization", "Context Sensitivity"], "observable": YES},
            {"name": "Context Dependency",
             "metrics": ["Context Gain", "Context Ablation Drop",
                         "Long-Range Dependency"], "observable": YES},
            {"name": "Context Position",
             "metrics": ["Position Sensitivity", "Lost-in-the-Middle Sensitivity"],
             "observable": YES},
            {"name": "Context Length",
             "metrics": ["Context-Length Sensitivity", "Long-Context Retention"],
             "observable": YES},
        ],
    },
    {
        "stage": 5,
        "key": "robustness",
        "layer": "Robustness",
        "purpose": "Measure resistance to prompt and input changes",
        "mmlu": YES, "lambada": YES, "openrouter": YES,
        "priority": P1,
        "requires": "robustness pass",
        "subsections": [
            {"name": "Prompt Robustness",
             "metrics": ["Prompt Variation Accuracy", "Prompt Stability"], "observable": YES},
            {"name": "Semantic Robustness",
             "metrics": ["Paraphrase Accuracy", "Paraphrase Consistency"], "observable": YES},
            {"name": "Input Perturbation",
             "metrics": ["Typographical", "Formatting", "Noise", "Case Robustness"],
             "observable": YES},
            {"name": "MMLU Choice Robustness",
             "metrics": ["Option-Order Robustness", "Answer-Position Bias"],
             "observable": YES, "benchmarks": ["mmlu"]},
            {"name": "Distribution Robustness",
             "metrics": ["OOD Accuracy", "OOD Perplexity"],
             "observable": YES,
             "note": "uses the LAMBADA control/rejected splits as the OOD set"},
        ],
    },
    {
        "stage": 6,
        "key": "api_performance",
        "layer": "API Performance",
        "purpose": "Measure inference-service performance",
        "mmlu": YES, "lambada": YES, "openrouter": YES,
        "priority": P0,
        "subsections": [
            {"name": "Initial Latency", "metrics": ["TTFT"], "observable": YES},
            {"name": "Generation Latency", "metrics": ["TPOT"], "observable": YES},
            {"name": "End-to-End Latency", "metrics": ["E2E Latency"], "observable": YES},
            {"name": "Latency Distribution", "metrics": ["P50", "P95", "P99"], "observable": YES},
            {"name": "Throughput",
             "metrics": ["Output Tokens/sec", "Prompt Tokens/sec", "Total Tokens/sec"],
             "observable": YES},
            {"name": "Request Throughput",
             "metrics": ["Requests/sec", "Items/sec"], "observable": YES},
        ],
    },
    {
        "stage": 7,
        "key": "token_efficiency",
        "layer": "Token Efficiency",
        "purpose": "Measure token consumption and efficiency",
        "mmlu": YES, "lambada": YES, "openrouter": YES,
        "priority": P0,
        "subsections": [
            {"name": "Input Usage", "metrics": ["Prompt Tokens"], "observable": YES},
            {"name": "Output Usage", "metrics": ["Completion Tokens"], "observable": YES},
            {"name": "Reasoning Usage", "metrics": ["Reasoning Tokens"],
             "observable": YES,
             "note": "returned in completion_tokens_details; 0 for non-reasoning models"},
            {"name": "Cached Input", "metrics": ["Cached Prompt Tokens"],
             "observable": YES,
             "note": "returned in prompt_tokens_details when the provider supports caching"},
            {"name": "Aggregate Usage",
             "metrics": ["Total Tokens", "Tokens/Question", "Tokens/Correct Answer"],
             "observable": YES},
        ],
    },
    {
        "stage": 8,
        "key": "economics",
        "layer": "Economics",
        "purpose": "Measure monetary efficiency",
        "mmlu": YES, "lambada": YES, "openrouter": YES,
        "priority": P0,
        "subsections": [
            {"name": "Request Cost", "metrics": ["Cost/Request"], "observable": YES},
            {"name": "Token Cost", "metrics": ["Cost/1K Tokens", "Cost/1M Tokens"],
             "observable": YES},
            {"name": "Quality-Adjusted Cost",
             "metrics": ["Cost/Correct Answer", "Quality per Dollar"], "observable": YES},
        ],
    },
    {
        "stage": 9,
        "key": "reliability",
        "layer": "Reliability",
        "purpose": "Measure operational stability",
        "mmlu": YES, "lambada": YES, "openrouter": YES,
        "priority": P0,
        "subsections": [
            {"name": "API Reliability",
             "metrics": ["Failure Rate", "API Error Rate", "Timeout Rate",
                         "Rate-Limit (429) Rate"], "observable": YES},
            {"name": "Output Reliability", "metrics": ["Invalid Output Rate"],
             "observable": YES},
            {"name": "Reproducibility",
             "metrics": ["Run Variance", "Seed Variance", "Reproducibility Score"],
             "observable": YES},
            {"name": "Operational Stability",
             "metrics": ["Retry Rate", "Provider Failover Count"],
             "observable": YES,
             "note": "retries and provider identity are visible client-side; "
                     "upstream OOM is not"},
        ],
    },
]


# Stage 10 is intentionally excluded from the OpenRouter benchmark. Kept here
# so the report can state precisely what is out of scope and why, rather than
# leaving a reader to wonder whether it was forgotten.
EXCLUDED_STAGE = {
    "stage": 10,
    "key": "hardware_distributed",
    "layer": "Hardware & Distributed",
    "purpose": "Measure underlying inference infrastructure",
    "mmlu": NO, "lambada": NO, "openrouter": NO,
    "reason": (
        "The benchmark is a client of a shared, auto-scaled third-party "
        "endpoint. Parallelism layout, GPU telemetry and power draw are chosen "
        "and held by the provider; none of them is exposed to the caller. Any "
        "figure reported here would be fabricated."
    ),
    "path_to_measure": (
        "Run the same suite against a self-hosted vLLM backend "
        "(benchmark_config.local_vllm_config), where TP/PP/DP/SP/CP/EP become "
        "settable variables and VRAM, KV cache, communication overhead and "
        "energy become directly measurable."
    ),
    "subsections": [
        {"name": "Tensor Parallelism", "metrics": ["TP Degree", "TP Speedup"]},
        {"name": "Pipeline Parallelism", "metrics": ["PP Degree", "PP Speedup"]},
        {"name": "Data/Sequence/Context/Expert Parallelism",
         "metrics": ["DP", "SP", "CP", "EP"]},
        {"name": "Hardware Telemetry",
         "metrics": ["GPU Utilization", "VRAM", "KV Cache", "Memory Bandwidth"]},
        {"name": "Compute Efficiency", "metrics": ["FLOPs", "MFU", "HFU"]},
        {"name": "Distributed Scaling",
         "metrics": ["Communication Overhead", "Parallel Efficiency", "Pipeline Bubble"]},
        {"name": "Energy", "metrics": ["Joules/token", "Tokens/Watt", "GPU Power"]},
    ],
}


# Presentation order and labelling for the UI.
STAGE_UI = {
    "task_quality":    {"icon": "🎯", "short": "Was the answer right?"},
    "probability":     {"icon": "📈", "short": "Was its confidence earned?"},
    "consistency":     {"icon": "🔁", "short": "Would it answer the same way again?"},
    "context":         {"icon": "📖", "short": "Is it actually reading the passage?"},
    "robustness":      {"icon": "🛡️", "short": "Does the score survive input changes?"},
    "api_performance": {"icon": "⚡", "short": "How fast did the service respond?"},
    "token_efficiency":{"icon": "🔤", "short": "How many tokens did it consume?"},
    "economics":       {"icon": "💰", "short": "What did it cost?"},
    "reliability":     {"icon": "🧯", "short": "Did the calls actually succeed?"},
    "tokenization":    {"icon": "✂️", "short": "Is the tokenizer handicapping it?"},
}


def stage_for(key: str) -> Optional[Dict]:
    for s in STAGES:
        if s["key"] == key:
            return s
    return None


def stages_for_benchmark(benchmark: str) -> List[Dict]:
    """Stages that apply to a benchmark, in order."""
    out = []
    for s in STAGES:
        if s.get(benchmark) in (NO,):
            continue
        out.append(s)
    return out


def priority_summary() -> Dict[str, List[str]]:
    """Stage layers grouped by priority band, for the report and the UI."""
    bands: Dict[str, List[str]] = {P0: [], P1: [], P2: []}
    for s in STAGES:
        bands.setdefault(s["priority"], []).append(s["layer"])
    return bands


def observability_report() -> Dict:
    """Machine-readable statement of what this deployment can measure."""
    return {
        "deployment": "openrouter_hosted_api",
        "stages": [
            {
                "stage": s["stage"], "layer": s["layer"], "key": s["key"],
                "priority": s["priority"], "openrouter": s["openrouter"],
                "requires": s.get("requires"),
            }
            for s in STAGES
        ],
        "excluded": {
            "stage": EXCLUDED_STAGE["stage"],
            "layer": EXCLUDED_STAGE["layer"],
            "reason": EXCLUDED_STAGE["reason"],
            "path_to_measure": EXCLUDED_STAGE["path_to_measure"],
        },
    }
