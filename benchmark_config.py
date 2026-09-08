"""Benchmark configuration variables - the independent variables of a run.

WHY THIS FILE EXISTS SEPARATELY FROM THE METRICS
------------------------------------------------
TP, PP, DP, SP, CP and EP are *knobs you set*, not *scores you earn*. Folding
them into the result dictionary alongside accuracy invites a category error:
you end up "comparing" a parallelism degree against a Brier score, or ranking
models on a composite that silently mixes a hardware layout with a quality
measurement.

So the design is:

    RunConfig   ->  independent variables (what we chose)
    metrics/*   ->  dependent variables   (what we observed)
    analyse_sweep()  ->  the relationship between them

Every result file records the full RunConfig that produced it. Two results are
only comparable when their configs match on everything except the one variable
under study - `RunConfig.diff()` exists to make that checkable rather than
assumed.

A single run answers "how good is this model?".
A sweep answers "how does this knob change latency, memory, cost - and does it
touch quality at all?". The second question is the interesting one, and it is
unanswerable if the knobs were never recorded.
"""

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

SCHEMA_VERSION = "2.0"


# ===========================================================================
# Parallelism - the six degrees, as configuration
# ===========================================================================

@dataclass
class ParallelismConfig:
    """How the model is sharded across devices.

    These are *inputs*. On a hosted API they are unknown and stay at their
    defaults with `controlled=False`; on a local vLLM/TensorRT-LLM backend
    they are set by the operator and become sweepable.

    tensor_parallel (TP)
        Split each weight matrix across GPUs. Every layer needs an all-reduce,
        so TP is latency-friendly but communication-heavy - best within one
        node over NVLink.
    pipeline_parallel (PP)
        Put different layers on different GPUs. Cheap communication (activations
        only, point-to-point) but introduces pipeline bubbles that hurt latency
        at small batch sizes.
    data_parallel (DP)
        Full model replicas serving different requests. Multiplies throughput,
        leaves per-request latency alone, multiplies memory.
    sequence_parallel (SP)
        Shard the sequence dimension in the norm/dropout regions that TP leaves
        replicated. Saves activation memory, usually paired with TP.
    context_parallel (CP)
        Shard the sequence dimension of attention itself (ring/Ulysses
        attention). The lever for very long contexts.
    expert_parallel (EP)
        Distribute MoE experts across devices. Inert for dense models - all
        three models here are dense, so EP stays 1 and is recorded to make
        that explicit rather than absent.
    """
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    sequence_parallel: int = 1
    context_parallel: int = 1
    expert_parallel: int = 1

    # False when we are a client of someone else's endpoint and cannot set these.
    controlled: bool = False
    note: str = ""

    @property
    def world_size(self) -> int:
        """Total devices implied by the layout."""
        return (self.tensor_parallel * self.pipeline_parallel
                * self.data_parallel * self.context_parallel)

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def label(self) -> str:
        """Compact identifier for charts and result filenames."""
        return (f"TP{self.tensor_parallel}_PP{self.pipeline_parallel}"
                f"_DP{self.data_parallel}_SP{self.sequence_parallel}"
                f"_CP{self.context_parallel}_EP{self.expert_parallel}")

    def validate(self) -> List[str]:
        """Return a list of configuration problems (empty when sane)."""
        problems = []
        for name in ("tensor_parallel", "pipeline_parallel", "data_parallel",
                     "sequence_parallel", "context_parallel", "expert_parallel"):
            if getattr(self, name) < 1:
                problems.append(f"{name} must be >= 1")
        if self.sequence_parallel > 1 and self.tensor_parallel == 1:
            problems.append(
                "sequence_parallel > 1 without tensor_parallel has no effect: "
                "SP shards the regions TP leaves replicated"
            )
        return problems


# ===========================================================================
# Serving / decoding / dataset configuration
# ===========================================================================

@dataclass
class ServingConfig:
    """Runtime knobs of the inference server (or of our client, when hosted)."""
    backend: str = "openrouter"          # openrouter | vllm | tgi | local
    dtype: str = "unknown"               # bf16 | fp16 | fp8 | int8 | int4
    quantization: Optional[str] = None
    kv_cache_dtype: str = "unknown"
    max_batch_size: Optional[int] = None
    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    enable_prefix_caching: Optional[bool] = None
    speculative_decoding: bool = False
    device: str = "remote"
    device_tdp_watts: Optional[float] = None
    controlled: bool = False


@dataclass
class DecodingConfig:
    """Sampling parameters. These DO affect quality, deliberately."""
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_tokens: int = 384
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    few_shot: int = 0


@dataclass
class DatasetConfig:
    """What was evaluated, so a result is reproducible."""
    benchmark: str = "mmlu"              # mmlu | lambada
    split: str = "test"
    subjects: List[str] = field(default_factory=list)
    n_items: int = 0
    questions_per_subject: Optional[int] = None
    sample_seed: int = 42
    prompt_template: str = "baseline"


@dataclass
class EvaluationConfig:
    """Which optional metric studies to run.

    All default False: they multiply API calls, and a run that silently costs
    5x is a bad default. The report states which were enabled.
    """
    request_logprobs: bool = True
    top_logprobs: int = 5
    stream_for_latency: bool = True
    repeats_for_consistency: int = 1     # >1 enables answer stability
    seeds_for_stability: List[int] = field(default_factory=list)
    robustness_variants: List[str] = field(default_factory=list)
    context_ablations: List[str] = field(default_factory=list)
    calibration_bins: int = 10


# ===========================================================================
# The run manifest
# ===========================================================================

@dataclass
class RunConfig:
    """Complete, serialisable description of one benchmark run.

    Stamped into every result file. This is what makes a number from March
    comparable - or provably not comparable - with a number from September.
    """
    run_id: str = ""
    timestamp: str = ""
    schema_version: str = SCHEMA_VERSION
    model: str = ""

    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    environment: Dict = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.run_id:
            safe_model = (self.model or "model").replace("/", "_")
            self.run_id = (
                f"{safe_model}_{self.dataset.benchmark}"
                f"_{self.parallelism.label()}"
                f"_{time.strftime('%Y%m%d-%H%M%S')}"
            )
        if not self.environment:
            self.environment = capture_environment()

    def to_dict(self) -> Dict:
        return asdict(self)

    def config_key(self) -> str:
        """Identity of everything except the model.

        Two runs sharing a config_key differ only in which model was asked,
        which is exactly the condition for a fair model-vs-model comparison.
        """
        return json.dumps({
            "parallelism": asdict(self.parallelism),
            "serving": asdict(self.serving),
            "decoding": asdict(self.decoding),
            "dataset": asdict(self.dataset),
        }, sort_keys=True)

    def diff(self, other: "RunConfig") -> Dict[str, Dict]:
        """Fields where two configs disagree.

        Use before comparing results: if the diff contains anything other than
        the variable being studied, the comparison is confounded.
        """
        out: Dict[str, Dict] = {}
        for section in ("parallelism", "serving", "decoding", "dataset", "evaluation"):
            a = asdict(getattr(self, section))
            b = asdict(getattr(other, section))
            changed = {
                k: {"this": a[k], "other": b[k]}
                for k in a if a.get(k) != b.get(k)
            }
            if changed:
                out[section] = changed
        return out

    def validate(self) -> List[str]:
        problems = list(self.parallelism.validate())
        if self.decoding.temperature > 0 and self.evaluation.repeats_for_consistency <= 1:
            problems.append(
                "temperature > 0 with a single repeat: results will be "
                "irreproducible and answer stability cannot be measured"
            )
        if self.parallelism.is_distributed and not self.parallelism.controlled:
            problems.append(
                "parallelism degrees > 1 recorded but controlled=False: "
                "a hosted API run cannot claim a known parallelism layout"
            )
        return problems


def capture_environment() -> Dict:
    """Snapshot of the machine running the harness (the client, when hosted)."""
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if out.returncode == 0:
            env["git_commit"] = out.stdout.strip()
    except Exception:
        pass
    return env


# ===========================================================================
# Presets
# ===========================================================================

def hosted_api_config(model: str, benchmark: str = "mmlu", **overrides) -> RunConfig:
    """Config for the deployment this project actually uses.

    Parallelism stays at 1 with controlled=False - not because the provider
    runs on one GPU (it almost certainly does not), but because we do not know
    what it runs on. Recording 1/False is the honest encoding of "unknown".
    """
    cfg = RunConfig(
        model=model,
        parallelism=ParallelismConfig(
            controlled=False,
            note=("hosted endpoint: layout chosen by the provider, not "
                  "observable or settable from the client"),
        ),
        serving=ServingConfig(
            backend="openrouter", device="remote", controlled=False,
        ),
        dataset=DatasetConfig(benchmark=benchmark),
    )
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def local_vllm_config(
    model: str,
    tp: int = 1, pp: int = 1, dp: int = 1,
    sp: int = 1, cp: int = 1, ep: int = 1,
    benchmark: str = "mmlu",
    device_tdp_watts: float = 300.0,
    **overrides,
) -> RunConfig:
    """Config for a local vLLM run, where the six degrees become real knobs.

    This is the path that turns the currently-unobservable systems metrics
    (VRAM, KV cache, energy, communication overhead) into measured ones and
    makes a parallelism sweep possible.
    """
    cfg = RunConfig(
        model=model,
        parallelism=ParallelismConfig(
            tensor_parallel=tp, pipeline_parallel=pp, data_parallel=dp,
            sequence_parallel=sp, context_parallel=cp, expert_parallel=ep,
            controlled=True,
            note="local vLLM: layout set by the operator and swept",
        ),
        serving=ServingConfig(
            backend="vllm", dtype="bf16", kv_cache_dtype="auto",
            device="cuda", device_tdp_watts=device_tdp_watts, controlled=True,
        ),
        dataset=DatasetConfig(benchmark=benchmark),
    )
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


# ===========================================================================
# Sweeps: varying one config variable and measuring the effect
# ===========================================================================

def build_parallelism_sweep(
    base: RunConfig,
    variable: str,
    values: Sequence[int],
) -> List[RunConfig]:
    """One RunConfig per value of a single parallelism degree.

    Deliberately one-at-a-time: with everything else pinned, any change in the
    dependent metrics is attributable to this variable. Grid sweeps are
    possible but need a factorial analysis this project does not have the
    sample budget for.
    """
    if not hasattr(ParallelismConfig(), variable):
        raise ValueError(
            f"unknown parallelism variable {variable!r}; expected one of "
            "tensor_parallel, pipeline_parallel, data_parallel, "
            "sequence_parallel, context_parallel, expert_parallel"
        )

    configs = []
    for value in values:
        par = ParallelismConfig(**{**asdict(base.parallelism)})
        setattr(par, variable, value)
        cfg = RunConfig(
            model=base.model,
            parallelism=par,
            serving=ServingConfig(**asdict(base.serving)),
            decoding=DecodingConfig(**asdict(base.decoding)),
            dataset=DatasetConfig(**asdict(base.dataset)),
            evaluation=EvaluationConfig(**asdict(base.evaluation)),
            notes=f"sweep: {variable}={value}",
        )
        configs.append(cfg)
    return configs


# Dependent variables a parallelism sweep is expected to move, split by
# whether the effect should be mechanical or (ideally) absent.
SYSTEMS_EFFECT_METRICS = [
    "latency.ttft.mean", "latency.tpot.mean", "latency.e2e.mean",
    "throughput.prefill_tokens_per_s", "throughput.decode_tokens_per_s",
    "throughput.requests_per_s",
    "memory.estimated_total_gb", "memory.kv_cache_gb",
    "energy.wh", "cost.total_usd",
]

QUALITY_INVARIANT_METRICS = [
    "overall_accuracy", "macro_accuracy_subject", "last_word_accuracy",
    "nll", "ece", "brier_score",
]


def analyse_sweep(runs: Sequence[Dict]) -> Dict:
    """Quantify how a swept config variable moved the observed metrics.

    `runs` is a list of {"config": RunConfig|dict, "metrics": {...}}.

    Two questions, deliberately asked separately:

      1. Systems  - did latency/throughput/memory respond, and how efficiently?
         Scaling efficiency is speedup / device-count-ratio; anything well
         below 1.0 is communication overhead eating the gain.

      2. Quality  - did accuracy move at all? It *should not*: parallelism
         changes arithmetic order, not model semantics. A quality shift beyond
         numerical noise is a red flag - a bug, or non-determinism worth
         reporting, not a "result".
    """
    if len(runs) < 2:
        return {"available": False, "reason": "a sweep needs at least two runs"}

    rows = []
    for r in runs:
        cfg = r["config"]
        cfg_d = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
        par = cfg_d.get("parallelism", {})
        rows.append({
            "run_id": cfg_d.get("run_id"),
            "label": _par_label(par),
            "world_size": (par.get("tensor_parallel", 1)
                           * par.get("pipeline_parallel", 1)
                           * par.get("data_parallel", 1)
                           * par.get("context_parallel", 1)),
            "parallelism": par,
            "metrics": r.get("metrics", {}),
        })

    rows.sort(key=lambda x: x["world_size"])
    baseline = rows[0]

    # Which parallelism fields actually differ across the sweep?
    varied = sorted({
        k for k in baseline["parallelism"]
        if isinstance(baseline["parallelism"].get(k), int)
        and len({r["parallelism"].get(k) for r in rows}) > 1
    })

    systems_effects = {}
    for path in SYSTEMS_EFFECT_METRICS:
        series = [(r["label"], _dig(r["metrics"], path)) for r in rows]
        series = [(l, v) for l, v in series if isinstance(v, (int, float))]
        if len(series) < 2:
            continue
        base_val = series[0][1]
        systems_effects[path] = {
            "series": [{"config": l, "value": v} for l, v in series],
            "baseline": base_val,
            "final": series[-1][1],
            "relative_change": round(
                (series[-1][1] - base_val) / base_val, 4
            ) if base_val else None,
        }

    # Scaling efficiency on throughput.
    scaling = []
    base_tp = _dig(baseline["metrics"], "throughput.decode_tokens_per_s")
    for r in rows:
        val = _dig(r["metrics"], "throughput.decode_tokens_per_s")
        if not (isinstance(val, (int, float)) and isinstance(base_tp, (int, float)) and base_tp):
            continue
        device_ratio = r["world_size"] / max(baseline["world_size"], 1)
        speedup = val / base_tp
        scaling.append({
            "config": r["label"],
            "world_size": r["world_size"],
            "speedup": round(speedup, 4),
            "ideal_speedup": round(device_ratio, 4),
            "scaling_efficiency": round(speedup / device_ratio, 4) if device_ratio else None,
            "communication_overhead": round(
                1 - speedup / device_ratio, 4
            ) if device_ratio else None,
        })

    # Quality should be flat. Flag it loudly if it is not.
    quality_drift = {}
    for path in QUALITY_INVARIANT_METRICS:
        vals = [_dig(r["metrics"], path) for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if len(vals) < 2:
            continue
        spread = max(vals) - min(vals)
        quality_drift[path] = {
            "values": vals,
            "spread": round(spread, 5),
            # 0.5pp is a pragmatic threshold: below it, batching
            # non-determinism explains the difference.
            "unexpected": bool(spread > 0.005),
        }

    flagged = [k for k, v in quality_drift.items() if v["unexpected"]]

    return {
        "available": True,
        "n_runs": len(rows),
        "varied_parallelism_fields": varied,
        "configurations": [
            {"label": r["label"], "world_size": r["world_size"]} for r in rows
        ],
        "systems_effects": systems_effects,
        "scaling": scaling,
        "quality_drift": quality_drift,
        "quality_drift_flagged": flagged,
        "interpretation": (
            "Systems metrics are expected to move with parallelism; quality "
            "metrics are expected to stay flat. "
            + (f"Quality drift exceeded tolerance for: {', '.join(flagged)} - "
               "investigate before reporting these as an effect of parallelism."
               if flagged else
               "No quality metric drifted beyond numerical tolerance, which is "
               "the expected and correct outcome.")
        ),
    }


def _par_label(par: Dict) -> str:
    return (f"TP{par.get('tensor_parallel', 1)}"
            f"_PP{par.get('pipeline_parallel', 1)}"
            f"_DP{par.get('data_parallel', 1)}"
            f"_SP{par.get('sequence_parallel', 1)}"
            f"_CP{par.get('context_parallel', 1)}"
            f"_EP{par.get('expert_parallel', 1)}")


def _dig(data: Dict, dotted: str):
    """Fetch a nested value by dotted path, tolerating gaps."""
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# ===========================================================================
# Model architecture facts (for the analytic memory model only)
# ===========================================================================

# Public architecture numbers, used solely by systems.estimate_memory().
# They describe the models, NOT the deployment we measured - any figure
# derived from them is labelled analytic_model.
MODEL_ARCH = {
    "google/gemma-3-4b-it": {
        "n_params": 4.3e9, "n_layers": 34, "n_heads": 8,
        "n_kv_heads": 4, "head_dim": 256, "vocab_size": 262144,
        "context_window": 131072,
    },
    "meta-llama/llama-3.2-3b-instruct": {
        "n_params": 3.21e9, "n_layers": 28, "n_heads": 24,
        "n_kv_heads": 8, "head_dim": 128, "vocab_size": 128256,
        "context_window": 131072,
    },
    "mistralai/ministral-8b-2512": {
        "n_params": 8.02e9, "n_layers": 36, "n_heads": 32,
        "n_kv_heads": 8, "head_dim": 128, "vocab_size": 131072,
        "context_window": 32768,
    },
}
