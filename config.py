import os
from dotenv import load_dotenv

# Load .env from this file's directory so it works regardless of the
# process working directory (e.g. when served under Passenger).
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Retry/backoff for 429 (rate limit) and 5xx responses from OpenRouter.
# Retry-After header is honored when present; otherwise exponential backoff
# up to OPENROUTER_RETRY_MAX_DELAY, capped at OPENROUTER_MAX_RETRIES attempts.
OPENROUTER_MAX_RETRIES = 6
OPENROUTER_RETRY_BASE_DELAY = 2.0
OPENROUTER_RETRY_MAX_DELAY = 30.0

# Route to the provider currently serving each model fastest rather than the
# cheapest one. Price-sorted routing (OpenRouter's default) tends to land on
# whichever backend is busiest/cheapest, which is what actually causes most
# sustained 429s here - it is provider congestion, not an account limit.
OPENROUTER_PROVIDER_SORT = "throughput"

MODELS = [
    "google/gemma-3-4b-it",
    "meta-llama/llama-3.2-3b-instruct",
    "mistralai/ministral-8b-2512",
]

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 32
MAX_TOKENS_REASONING = 8096
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0
FEW_SHOT_COUNT = 3

NUM_SAMPLES = 100
MIN_SAMPLES = 1
MAX_SAMPLES = 1000

REASONING_MODELS = []

# Fine-tuning presets: Optimal favours accuracy (greedy decoding, most
# worked examples), Normal is the balanced default, and Best Performance
# trims the token/example budget for the fastest, cheapest runs.
PRESETS = {
    "optimal": {
        "label": "Optimal",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 16,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "few_shot": 5,
    },
    "normal": {
        "label": "Normal",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 32,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "few_shot": 3,
    },
    "best": {
        "label": "Best Performance",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 8,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "few_shot": 2,
    },
}

# MMLU decoding presets. Each names the trade-off it makes so the panel is
# self-explanatory: greedy/deterministic for benchmarking, sampled for
# diversity studies, capped budgets for cheap runs.
MMLU_PRESETS = {
    "optimal": {
        "label": "Optimal",
        "hint": "Greedy, most reasoning room. Most reliable accuracy.",
        "temperature": 0.0, "top_p": 1.0, "top_k": 0, "max_tokens": 512,
        "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
        "min_p": 0.0, "seed": 42,
    },
    "normal": {
        "label": "Normal",
        "hint": "Light sampling. Matches the saved-run defaults.",
        "temperature": 0.2, "top_p": 0.95, "top_k": 0, "max_tokens": 384,
        "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
        "min_p": 0.0, "seed": 42,
    },
    "best": {
        "label": "Fast / Cheap",
        "hint": "Capped reasoning budget. Fastest and cheapest per question.",
        "temperature": 0.0, "top_p": 1.0, "top_k": 0, "max_tokens": 192,
        "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
        "min_p": 0.0, "seed": 42,
    },
    "deterministic": {
        "label": "Deterministic",
        "hint": "Greedy with a fixed seed - for reproducibility and seed-stability studies.",
        "temperature": 0.0, "top_p": 1.0, "top_k": 1, "max_tokens": 384,
        "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
        "min_p": 0.0, "seed": 42,
    },
    "creative": {
        "label": "Exploratory",
        "hint": "Sampled. Use with repeats>1 to measure self-consistency.",
        "temperature": 0.8, "top_p": 0.9, "top_k": 40, "max_tokens": 512,
        "frequency_penalty": 0.1, "presence_penalty": 0.1, "repetition_penalty": 1.05,
        "min_p": 0.05, "seed": None,
    },
}

# Full decoding parameter surface exposed by the web panel. Each entry drives
# one control and its explanation, so adding a parameter here adds it to the UI
# without touching the template.
DECODING_PARAMS = [
    {"id": "temperature", "label": "Temperature", "min": 0.0, "max": 2.0, "step": 0.05,
     "default": 0.0, "type": "range",
     "hint": "Randomness. 0 = always take the most likely token (best for benchmarking)."},
    {"id": "top_p", "label": "Top-p (nucleus)", "min": 0.0, "max": 1.0, "step": 0.05,
     "default": 1.0, "type": "range",
     "hint": "Sample only from the smallest set of tokens whose probabilities sum to p."},
    {"id": "top_k", "label": "Top-k", "min": 0, "max": 100, "step": 1,
     "default": 0, "type": "range",
     "hint": "Consider only the k most likely tokens. 0 disables the limit."},
    {"id": "min_p", "label": "Min-p", "min": 0.0, "max": 1.0, "step": 0.01,
     "default": 0.0, "type": "range",
     "hint": "Drop tokens below this fraction of the top token's probability."},
    {"id": "max_tokens", "label": "Max tokens", "min": 8, "max": 1024, "step": 8,
     "default": 384, "type": "range",
     "hint": "Output budget. Too low truncates reasoning before the answer letter."},
    {"id": "frequency_penalty", "label": "Frequency penalty", "min": -2.0, "max": 2.0,
     "step": 0.1, "default": 0.0, "type": "range",
     "hint": "Penalises tokens by how often they have already appeared."},
    {"id": "presence_penalty", "label": "Presence penalty", "min": -2.0, "max": 2.0,
     "step": 0.1, "default": 0.0, "type": "range",
     "hint": "Penalises tokens that have appeared at all, regardless of count."},
    {"id": "repetition_penalty", "label": "Repetition penalty", "min": 0.5, "max": 2.0,
     "step": 0.05, "default": 1.0, "type": "range",
     "hint": "Divides the logit of already-seen tokens. 1.0 = off."},
    {"id": "seed", "label": "Seed", "min": 0, "max": 9999, "step": 1,
     "default": 42, "type": "number",
     "hint": "Fixed seed makes a run repeatable where the provider honours it. Blank = random."},
]

# Extra evaluation passes offered in the web panel. These map onto the
# run_benchmark.py flags of the same name.
EVAL_PASSES = [
    {"id": "calibration", "label": "Calibration pass", "default": True,
     "cost": "+1 call/item (1 token each)",
     "hint": "Single-token scoring to recover ECE, Brier, NLL and target rank. "
             "Only yields numbers for providers that return log-probabilities."},
    {"id": "robustness", "label": "Robustness sweep", "default": False,
     "cost": "+2-4 calls/item",
     "hint": "Prompt variation, option reordering and typo noise, scored as accuracy deltas."},
    {"id": "context", "label": "Context ablations", "default": False,
     "cost": "+3 calls/item (LAMBADA)",
     "hint": "Re-runs with the passage cut down, to measure real context utilisation."},
    {"id": "repeats", "label": "Consistency repeats", "default": 1,
     "type": "number", "min": 1, "max": 5,
     "cost": "x(N-1) extra calls",
     "hint": "Repeat each item N times to measure answer stability and self-consistency."},
]

DATASET_DIR = "_rsc/lambada-dataset"
REJECTED_DIR = "_rsc/rejected-data1/rejected"
RESULTS_DIR = "results"
DIAGRAMS_DIR = "diagrams"
REPORT_DIR = "report"

DATASET_FILES = {
    "test": os.path.join(DATASET_DIR, "lambada_test_plain_text.txt"),
    "development": os.path.join(DATASET_DIR, "lambada_development_plain_text.txt"),
    "control_test": os.path.join(DATASET_DIR, "lambada_control_test_data_plain_text.txt"),
    "rejected": os.path.join(REJECTED_DIR, "rejected_plain_text.txt"),
}

MODEL_INFO = {
    "google/gemma-3-4b-it": {
        "name": "Gemma-3-4B",
        "developer": "Google",
        "params": "4B",
        "architecture": "Dense decoder-only transformer with interleaved local/global attention",
        "technique": "Knowledge distillation + local/global attention interleaving",
        "description": (
            "Gemma-3-4B is a 4 billion parameter instruction-tuned model from Google, built "
            "with the same research that powers Gemini. It is a dense decoder-only transformer "
            "that interleaves several local sliding-window attention layers with an occasional "
            "global attention layer, which keeps memory low while still letting information "
            "flow across a long context (up to 128k tokens). The small Gemma 3 models are "
            "trained with knowledge distillation from larger teacher models."
        ),
        "technique_detail": (
            "Gemma 3 alternates attention types across depth: most layers only look at a "
            "local window of nearby tokens (cheap), and every few layers one global layer "
            "lets any token attend to the whole context (expressive). Combined with grouped "
            "query attention and QK-norm for stable training, and with distillation - the "
            "small model learns to match a larger teacher's output distribution rather than "
            "raw text alone - it delivers strong quality per parameter at edge-friendly cost."
        ),
        "key_properties": [
            "Dense decoder-only transformer, no expert routing.",
            "5:1 interleaving of local sliding-window and global attention layers.",
            "Grouped Query Attention with QK-norm; 128k-token context window.",
            "Distilled from larger Gemma/Gemini-family teacher models.",
        ],
        "flow_mermaid": (
            "graph TD\n"
            "    A[Input Tokens] --> B[Token Embedding + RoPE]\n"
            "    B --> C[Decoder Layer x N]\n"
            "    C --> D[Local Sliding-Window Attention x5]\n"
            "    D --> E[Global Attention x1]\n"
            "    E --> F[GeGLU Feed Forward]\n"
            "    F --> G[RMSNorm + Residual]\n"
            "    G --> C\n"
            "    G --> H[Final RMSNorm]\n"
            "    H --> I[LM Head]\n"
            "    I --> J[Softmax to Next Token]"
        ),
        "strengths": "Strong quality per parameter, long context, cheap to serve",
        "weaknesses": "Global reasoning can lag models with full attention everywhere",
        "color": "#2196F3",
    },
    "meta-llama/llama-3.2-3b-instruct": {
        "name": "Llama-3.2-3B",
        "developer": "Meta",
        "params": "3B",
        "architecture": "Dense decoder-only transformer with Grouped Query Attention",
        "technique": "Compact dense transformer",
        "description": (
            "Llama-3.2-3B is a 3 billion parameter instruction-tuned model from Meta, aimed at "
            "on-device and low-cost deployment. It follows the standard Llama recipe: a dense "
            "decoder-only transformer with rotary position embeddings, grouped query attention, "
            "and SwiGLU feed-forward layers. The smaller Llama 3.2 models were partly built by "
            "pruning and distilling from larger Llama 3.1 models."
        ),
        "technique_detail": (
            "Llama-3.2-3B keeps the well-tested dense transformer design and makes it small. "
            "Rotary embeddings encode position, grouped query attention shrinks the key-value "
            "cache, and SwiGLU activations improve the feed-forward layers. The result is a "
            "predictable, easy-to-serve model that runs comfortably on modest hardware."
        ),
        "key_properties": [
            "Dense decoder-only transformer.",
            "Rotary position embeddings (RoPE) for length generalisation.",
            "Grouped Query Attention and SwiGLU feed-forward layers.",
            "Distilled and pruned from larger Llama 3.1 checkpoints.",
        ],
        "flow_mermaid": (
            "graph TD\n"
            "    A[Input Tokens] --> B[Token Embedding + RoPE]\n"
            "    B --> C[Decoder Layer x N]\n"
            "    C --> D[Multi-Head Attention with GQA]\n"
            "    D --> E[SwiGLU Feed Forward]\n"
            "    E --> F[RMSNorm + Residual]\n"
            "    F --> C\n"
            "    F --> G[Final RMSNorm]\n"
            "    G --> H[LM Head]\n"
            "    H --> I[Softmax to Next Token]"
        ),
        "strengths": "Very small, broad ecosystem support, runs on-device",
        "weaknesses": "Lower ceiling on complex reasoning tasks",
        "color": "#4CAF50",
    },
    "mistralai/ministral-8b-2512": {
        "name": "Ministral-8B",
        "developer": "Mistral AI",
        "params": "8B",
        "architecture": "Decoder-only transformer with Sliding Window Attention",
        "technique": "Sliding Window Attention + GQA",
        "description": (
            "Ministral-8B is an 8 billion parameter model from Mistral AI, part of the Ministral "
            "family built for edge use. It uses interleaved sliding window attention so each "
            "layer only attends to a local window of recent tokens, which keeps memory and "
            "compute low on long inputs. Grouped query attention further reduces the key-value "
            "cache, and the model handles long contexts efficiently."
        ),
        "technique_detail": (
            "Standard attention compares every token with every other token, which grows "
            "quadratically with length. Sliding window attention limits each layer to a fixed "
            "window of recent tokens, and stacking layers lets information travel further than "
            "any single window. Grouped query attention shares key-value heads to cut memory, "
            "giving fast decoding on long passages."
        ),
        "key_properties": [
            "Interleaved sliding window attention for local context.",
            "Grouped Query Attention for a smaller key-value cache.",
            "Deep stacking propagates context beyond a single window.",
            "Tuned for efficient long-context inference at the edge.",
        ],
        "flow_mermaid": (
            "graph TD\n"
            "    A[Input Tokens] --> B[Token Embedding + RoPE]\n"
            "    B --> C[Decoder Layer x N]\n"
            "    C --> D[Sliding Window Attention]\n"
            "    D --> E[Grouped Query Attention]\n"
            "    E --> F[SwiGLU Feed Forward]\n"
            "    F --> G[RMSNorm + Residual]\n"
            "    G --> C\n"
            "    G --> H[Final RMSNorm]\n"
            "    H --> I[LM Head]\n"
            "    I --> J[Softmax to Next Token]"
        ),
        "strengths": "Efficient long context, balanced size and quality",
        "weaknesses": "Distant context can fade across windows",
        "color": "#FF9800",
    },
}
