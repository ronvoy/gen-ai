import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    "x-ai/grok-3-mini",
    "mistralai/ministral-14b-2512",
    "deepseek/deepseek-r1-distill-qwen-32b",
]

TEMPERATURE = 0.0
MAX_TOKENS = 10
NUM_SAMPLES = 100

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
    "x-ai/grok-3-mini": {
        "name": "Grok-3-Mini",
        "developer": "xAI",
        "params": "~3B (estimated)",
        "architecture": "Transformer decoder-only with Mixture of Experts (MoE)",
        "description": (
            "Grok-3-Mini is a compact reasoning model from xAI designed for fast inference "
            "while retaining strong logical and analytical capabilities. It leverages a Mixture "
            "of Experts architecture to activate only a subset of parameters per token, achieving "
            "high throughput. The model is trained with reinforcement learning from human feedback "
            "(RLHF) and excels at chain-of-thought reasoning, code generation, and structured tasks."
        ),
        "strengths": "Fast inference, strong reasoning, efficient MoE routing, good at code and logic",
        "weaknesses": "Smaller capacity may limit performance on knowledge-intensive tasks",
        "color": "#4CAF50",
    },
    "mistralai/ministral-14b-2512": {
        "name": "Ministral-14B",
        "developer": "Mistral AI",
        "params": "14B",
        "architecture": "Transformer decoder-only with Sliding Window Attention (SWA)",
        "description": (
            "Ministral-14B is a 14-billion parameter model from Mistral AI, part of the Ministral "
            "family optimized for edge deployment and efficient inference. It uses Sliding Window "
            "Attention to handle long contexts efficiently, combined with Grouped Query Attention "
            "(GQA) for reduced memory footprint. The model supports multiple languages and is "
            "fine-tuned for instruction following with strong performance across general NLP tasks."
        ),
        "strengths": "Multilingual support, efficient attention, balanced size-performance trade-off",
        "weaknesses": "Mid-range size may underperform larger models on complex reasoning",
        "color": "#2196F3",
    },
    "deepseek/deepseek-r1-distill-qwen-32b": {
        "name": "DeepSeek-R1-Distill-Qwen-32B",
        "developer": "DeepSeek",
        "params": "32B",
        "architecture": "Qwen-based transformer with distilled reasoning from DeepSeek-R1",
        "description": (
            "DeepSeek-R1-Distill-Qwen-32B is a 32-billion parameter model created by distilling "
            "the reasoning capabilities of the full DeepSeek-R1 model into the Qwen-2.5 architecture. "
            "This knowledge distillation process transfers advanced chain-of-thought reasoning patterns "
            "into a more compact model. It inherits Qwen's efficient transformer design with RoPE "
            "positional embeddings and SwiGLU activations, while gaining DeepSeek-R1's strong "
            "multi-step reasoning and mathematical problem-solving abilities."
        ),
        "strengths": "Strong reasoning via distillation, large context window, excellent at complex tasks",
        "weaknesses": "Largest model of the three, slower inference, higher API cost",
        "color": "#FF9800",
    },
}
