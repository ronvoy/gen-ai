import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    "microsoft/phi-4-mini-instruct",
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
    "balanced": {
        "label": "Balanced",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 32,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "few_shot": 3,
    },
    "creative": {
        "label": "Creative",
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 48,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.3,
        "few_shot": 1,
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
    "microsoft/phi-4-mini-instruct": {
        "name": "Phi-4-Mini",
        "developer": "Microsoft",
        "params": "3.8B",
        "architecture": "Dense decoder-only transformer with Grouped Query Attention",
        "technique": "Curated and synthetic data training",
        "description": (
            "Phi-4-Mini is a 3.8 billion parameter model from Microsoft. The Phi family is "
            "built around data quality rather than raw scale: it is trained on filtered web "
            "data and synthetic textbook-style material, which lets a small model punch above "
            "its weight on reasoning and language tasks. It is a dense decoder-only transformer "
            "with grouped query attention and a 128k token vocabulary."
        ),
        "technique_detail": (
            "The Phi approach focuses on the training data. Instead of scraping ever larger "
            "corpora, the team curates high-signal web text and generates synthetic examples "
            "that resemble textbook explanations. A smaller model trained on cleaner data "
            "learns more per parameter, so Phi-4-Mini stays cheap to run while remaining "
            "competitive on language understanding."
        ),
        "key_properties": [
            "Dense decoder-only transformer, no expert routing.",
            "Grouped Query Attention for a smaller key-value cache.",
            "Trained on curated web plus synthetic textbook-quality data.",
            "Small footprint suited to edge and low-latency serving.",
        ],
        "flow_mermaid": (
            "graph TD\n"
            "    A[Input Tokens] --> B[Token Embedding + RoPE]\n"
            "    B --> C[Decoder Layer x N]\n"
            "    C --> D[Grouped Query Attention]\n"
            "    D --> E[SwiGLU Feed Forward]\n"
            "    E --> F[RMSNorm + Residual]\n"
            "    F --> C\n"
            "    F --> G[Final RMSNorm]\n"
            "    G --> H[LM Head]\n"
            "    H --> I[Softmax to Next Token]"
        ),
        "strengths": "Strong reasoning for its size, fast, cheap to serve",
        "weaknesses": "Limited world knowledge compared to larger models",
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
