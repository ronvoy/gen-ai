import os
from dotenv import load_dotenv

# Load .env from this file's directory so it works regardless of the
# process working directory (e.g. when served under Passenger).
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

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

# Same three presets for the MMLU decoding panel. Optimal gives the model
# the most room to reason, Normal matches the defaults, Best Performance
# caps the reasoning budget so runs finish faster.
MMLU_PRESETS = {
    "optimal": {
        "label": "Optimal",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
    },
    "normal": {
        "label": "Normal",
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 384,
    },
    "best": {
        "label": "Best Performance",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 192,
    },
}

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
