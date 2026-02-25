"""
Diagram Generation Script

Produces matplotlib comparison charts and mermaid workflow diagrams
saved as PNG files inside the diagrams/ directory.
"""

import os
import json
import base64
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR, DIAGRAMS_DIR, MODELS, MODEL_INFO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(split="test"):
    path = os.path.join(RESULTS_DIR, f"summary_{split}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_detailed(model_id, split="test"):
    safe = model_id.replace("/", "_")
    path = os.path.join(RESULTS_DIR, f"{safe}_lambada_{split}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _short(model_id):
    return MODEL_INFO.get(model_id, {}).get("name", model_id.split("/")[-1])


def _colors():
    return [MODEL_INFO[m]["color"] for m in MODELS if m in MODEL_INFO]


# ---------------------------------------------------------------------------
# Mermaid -> PNG via mermaid.ink
# ---------------------------------------------------------------------------

def save_mermaid_as_png(mermaid_code, output_path):
    try:
        encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("utf-8")
        url = f"https://mermaid.ink/img/{encoded}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
        print(f"  Saved: {output_path}")
        return True
    except Exception as e:
        print(f"  Warning: mermaid.ink render failed ({e}), skipping {output_path}")
        return False


# ---------------------------------------------------------------------------
# Matplotlib charts
# ---------------------------------------------------------------------------

def generate_accuracy_chart(summary, output_path):
    models = [_short(m["model"]) for m in summary["models"]]
    accs = [m["accuracy"] * 100 for m in summary["models"]]
    colors = _colors()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accs, color=colors[:len(models)], width=0.5, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"LAMBADA Benchmark \u2014 Accuracy Comparison ({summary['split']} split)", fontsize=14)
    ax.set_ylim(0, max(accs) * 1.25 if accs else 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_response_time_chart(summary, output_path):
    models = [_short(m["model"]) for m in summary["models"]]
    times = [m["avg_response_time"] for m in summary["models"]]
    colors = _colors()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, times, color=colors[:len(models)], width=0.5, edgecolor="white")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
                f"{t:.3f}s", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Avg Response Time (s)", fontsize=12)
    ax.set_title(f"LAMBADA Benchmark \u2014 Response Time ({summary['split']} split)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_combined_chart(summary, output_path):
    models = [_short(m["model"]) for m in summary["models"]]
    accs = [m["accuracy"] * 100 for m in summary["models"]]
    times = [m["avg_response_time"] for m in summary["models"]]
    colors = _colors()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = ax1.bar(models, accs, color=colors[:len(models)], width=0.5)
    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                 f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy")
    ax1.set_ylim(0, max(accs) * 1.25 if accs else 100)
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(models, times, color=colors[:len(models)], width=0.5)
    for bar, t in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
                 f"{t:.3f}s", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylabel("Avg Response Time (s)")
    ax2.set_title("Response Time")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"LAMBADA Benchmark Comparison ({summary['split']} split)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_error_rate_chart(summary, output_path):
    models = [_short(m["model"]) for m in summary["models"]]
    error_rates = [m["errors"] / m["total"] * 100 if m["total"] else 0 for m in summary["models"]]
    colors = _colors()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, error_rates, color=colors[:len(models)], width=0.5, edgecolor="white")
    for bar, er in zip(bars, error_rates):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.1,
                f"{er:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Error Rate (%)", fontsize=12)
    ax.set_title(f"LAMBADA Benchmark \u2014 API Error Rate ({summary['split']} split)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_radar_chart(summary, output_path):
    """Spider / radar chart comparing all metrics across models."""
    models_data = summary["models"]
    names = [_short(m["model"]) for m in models_data]
    colors = _colors()

    max_acc = max(m["accuracy"] for m in models_data) or 1
    max_time = max(m["avg_response_time"] for m in models_data) or 1
    max_err = max((m["errors"] / m["total"] * 100 if m["total"] else 0) for m in models_data) or 1

    categories = ["Accuracy", "Speed\n(inv. time)", "Reliability\n(inv. errors)", "Correct\nCount"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, m in enumerate(models_data):
        acc_norm = m["accuracy"] / max_acc if max_acc else 0
        speed_norm = 1 - (m["avg_response_time"] / max_time) if max_time else 0
        speed_norm = max(speed_norm, 0.05)
        err_pct = m["errors"] / m["total"] * 100 if m["total"] else 0
        rel_norm = 1 - (err_pct / max_err) if max_err else 1
        rel_norm = max(rel_norm, 0.05)
        correct_norm = m["correct"] / m["total"] if m["total"] else 0

        values = [acc_norm, speed_norm, rel_norm, correct_norm]
        values += values[:1]

        ax.fill(angles, values, alpha=0.15, color=colors[i])
        ax.plot(angles, values, "o-", linewidth=2, color=colors[i], label=names[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Multi-Metric Radar Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_per_model_time_histogram(split="test"):
    """Histogram of per-sample response times for each model."""
    colors = _colors()
    for idx, model_id in enumerate(MODELS):
        detail = load_detailed(model_id, split)
        if not detail or not detail.get("results"):
            continue
        times = [r["time"] for r in detail["results"] if r["time"] > 0]
        if not times:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(times, bins=20, color=colors[idx], edgecolor="white", alpha=0.85)
        ax.set_xlabel("Response Time (s)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"{_short(model_id)} \u2014 Response Time Distribution", fontsize=14)
        ax.axvline(np.mean(times), color="red", linestyle="--", label=f"Mean {np.mean(times):.3f}s")
        ax.axvline(np.median(times), color="blue", linestyle="--", label=f"Median {np.median(times):.3f}s")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        safe = model_id.replace("/", "_")
        out = os.path.join(DIAGRAMS_DIR, f"time_hist_{safe}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Mermaid diagrams — project-level
# ---------------------------------------------------------------------------

def generate_workflow_diagram(output_path):
    mermaid = (
        "graph TD\n"
        "    A[Start: Load Configuration] --> B[Load LAMBADA Dataset]\n"
        "    B --> C[Sample N Test Passages]\n"
        "    C --> D{For Each Model}\n"
        "    D --> E1[Grok-3-Mini]\n"
        "    D --> E2[Ministral-14B]\n"
        "    D --> E3[DeepSeek-R1-Distill-32B]\n"
        "    E1 --> F1[OpenRouter API]\n"
        "    E2 --> F2[OpenRouter API]\n"
        "    E3 --> F3[OpenRouter API]\n"
        "    F1 --> G[Collect Predictions]\n"
        "    F2 --> G\n"
        "    F3 --> G\n"
        "    G --> H[Calculate Accuracy & Metrics]\n"
        "    H --> I[Save Results JSON]\n"
        "    I --> J[Generate Charts & Diagrams]\n"
        "    J --> K[Generate Report & Slides]\n"
        "    K --> L[Streamlit Dashboard]\n"
        "    style A fill:#e1f5fe\n"
        "    style L fill:#c8e6c9\n"
        "    style H fill:#fff3e0"
    )
    return save_mermaid_as_png(mermaid, output_path)


def generate_architecture_diagram(output_path):
    mermaid = (
        "graph LR\n"
        "    subgraph Input\n"
        "        A[LAMBADA Passage] --> B[Remove Last Word]\n"
        "        B --> C[Context Prompt]\n"
        "    end\n"
        "    subgraph OpenRouter_API\n"
        "        C --> D[Grok-3-Mini]\n"
        "        C --> E[Ministral-14B]\n"
        "        C --> F[DeepSeek-R1-Distill]\n"
        "    end\n"
        "    subgraph Evaluation\n"
        "        D --> G[Prediction]\n"
        "        E --> G\n"
        "        F --> G\n"
        "        G --> H{Exact Match?}\n"
        "        H --> I[Correct]\n"
        "        H --> J[Incorrect]\n"
        "    end\n"
        "    subgraph Metrics\n"
        "        I --> K[Accuracy]\n"
        "        J --> K\n"
        "        K --> L[Response Time]\n"
        "        L --> M[Final Report]\n"
        "    end\n"
        "    style A fill:#e3f2fd\n"
        "    style M fill:#e8f5e9"
    )
    return save_mermaid_as_png(mermaid, output_path)


def generate_evaluation_pipeline_diagram(output_path):
    mermaid = (
        "graph TD\n"
        "    A[LAMBADA Dataset] --> B[Test: 5153 passages]\n"
        "    A --> C[Dev: 4869 passages]\n"
        "    A --> D[Rejected Data]\n"
        "    B --> E[Random Sample N]\n"
        "    C --> E\n"
        "    E --> F[Extract Context + Target]\n"
        "    F --> G[Construct Prompt]\n"
        "    G --> H[OpenRouter API Call]\n"
        "    H --> I[Parse Response]\n"
        "    I --> J[Normalize: lowercase + strip]\n"
        "    J --> K[Exact Match Comparison]\n"
        "    K --> L[Accuracy]\n"
        "    K --> M[Per-sample Results]\n"
        "    H --> N[Response Latency]\n"
        "    L --> O[Summary JSON]\n"
        "    M --> O\n"
        "    N --> O\n"
        "    style A fill:#e8eaf6\n"
        "    style O fill:#e8f5e9\n"
        "    style K fill:#fff3e0"
    )
    return save_mermaid_as_png(mermaid, output_path)


def generate_lambada_task_diagram(output_path):
    mermaid = (
        "graph LR\n"
        "    A[\"Full Passage\"] --> B[\"Context: all words except last\"]\n"
        "    A --> C[\"Target: last word\"]\n"
        "    B --> D[Send to LLM]\n"
        "    D --> E[\"Predicted Word\"]\n"
        "    E --> F{Match?}\n"
        "    C --> F\n"
        "    F -->|Yes| G[Correct]\n"
        "    F -->|No| H[Incorrect]\n"
        "    style G fill:#c8e6c9\n"
        "    style H fill:#ffcdd2"
    )
    return save_mermaid_as_png(mermaid, output_path)


# ---------------------------------------------------------------------------
# Mermaid diagrams — per-model inference flow
# ---------------------------------------------------------------------------

def generate_grok_flow(output_path):
    mermaid = (
        "graph TD\n"
        "    A[Input Tokens] --> B[Token Embedding + RoPE Positional Encoding]\n"
        "    B --> C[Transformer Decoder Layer x N]\n"
        "    C --> D[Self-Attention with Causal Mask]\n"
        "    D --> E[MoE Router: select top-k experts]\n"
        "    E --> F1[Expert FFN 1]\n"
        "    E --> F2[Expert FFN 2]\n"
        "    E --> F3[Expert FFN k]\n"
        "    F1 --> G[Weighted Sum of Expert Outputs]\n"
        "    F2 --> G\n"
        "    F3 --> G\n"
        "    G --> H[Residual + LayerNorm]\n"
        "    H --> C\n"
        "    H --> I[Final LayerNorm]\n"
        "    I --> J[LM Head: Vocabulary Projection]\n"
        "    J --> K[Softmax -> Next Token]\n"
        "    style A fill:#e8f5e9\n"
        "    style E fill:#fff3e0\n"
        "    style K fill:#c8e6c9"
    )
    return save_mermaid_as_png(mermaid, output_path)


def generate_ministral_flow(output_path):
    mermaid = (
        "graph TD\n"
        "    A[Input Tokens] --> B[Token Embedding + RoPE Positional Encoding]\n"
        "    B --> C[Transformer Decoder Layer x N]\n"
        "    C --> D[Sliding Window Attention - SWA]\n"
        "    D --> E[Grouped Query Attention - GQA]\n"
        "    E --> F[SwiGLU Feed-Forward Network]\n"
        "    F --> G[RMSNorm + Residual Connection]\n"
        "    G --> C\n"
        "    G --> H[Final RMSNorm]\n"
        "    H --> I[LM Head: Vocabulary Projection]\n"
        "    I --> J[Softmax -> Next Token]\n"
        "    style A fill:#e3f2fd\n"
        "    style D fill:#bbdefb\n"
        "    style J fill:#c8e6c9"
    )
    return save_mermaid_as_png(mermaid, output_path)


def generate_deepseek_flow(output_path):
    mermaid = (
        "graph TD\n"
        "    A[Input Tokens] --> B[Token Embedding + RoPE Positional Encoding]\n"
        "    B --> C[Qwen-2.5 Transformer Decoder Layer x N]\n"
        "    C --> D[Multi-Head Self-Attention with Causal Mask]\n"
        "    D --> E[SwiGLU Feed-Forward Network]\n"
        "    E --> F[RMSNorm + Residual Connection]\n"
        "    F --> C\n"
        "    F --> G[Final RMSNorm]\n"
        "    G --> H[LM Head: Vocabulary Projection]\n"
        "    H --> I[Softmax -> Next Token]\n"
        "    subgraph Distillation_Origin\n"
        "        T1[DeepSeek-R1 Teacher] --> T2[Chain-of-Thought Reasoning Data]\n"
        "        T2 --> T3[Knowledge Distillation Loss]\n"
        "        T3 --> C\n"
        "    end\n"
        "    style A fill:#fff3e0\n"
        "    style T1 fill:#ffe0b2\n"
        "    style I fill:#c8e6c9"
    )
    return save_mermaid_as_png(mermaid, output_path)


def generate_benchmark_method_diagram(output_path):
    mermaid = (
        "graph TD\n"
        "    A[LAMBADA Benchmark Method] --> B[Exact-Match Accuracy]\n"
        "    A --> C[Response Latency]\n"
        "    A --> D[Error Rate]\n"
        "    A --> E[Throughput]\n"
        "    B --> B1[Normalize prediction: lowercase + strip punctuation]\n"
        "    B1 --> B2[Compare predicted word == target word]\n"
        "    B2 --> B3[Accuracy = correct / total]\n"
        "    C --> C1[Record wall-clock time per API call]\n"
        "    C1 --> C2[Compute mean, median, min, max latency]\n"
        "    D --> D1[Count API failures or timeouts]\n"
        "    D1 --> D2[Error Rate = errors / total]\n"
        "    E --> E1[Samples evaluated / total wall-clock time]\n"
        "    style A fill:#e8eaf6\n"
        "    style B3 fill:#c8e6c9\n"
        "    style C2 fill:#bbdefb\n"
        "    style D2 fill:#ffcdd2\n"
        "    style E1 fill:#fff3e0"
    )
    return save_mermaid_as_png(mermaid, output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_diagrams(split="test"):
    os.makedirs(DIAGRAMS_DIR, exist_ok=True)

    print("\nGenerating diagrams...")
    print("=" * 40)

    # Project-level mermaid diagrams
    generate_workflow_diagram(os.path.join(DIAGRAMS_DIR, "project_workflow.png"))
    generate_architecture_diagram(os.path.join(DIAGRAMS_DIR, "architecture_comparison.png"))
    generate_evaluation_pipeline_diagram(os.path.join(DIAGRAMS_DIR, "evaluation_pipeline.png"))
    generate_lambada_task_diagram(os.path.join(DIAGRAMS_DIR, "lambada_task.png"))
    generate_benchmark_method_diagram(os.path.join(DIAGRAMS_DIR, "benchmark_method.png"))

    # Per-model inference flow diagrams
    generate_grok_flow(os.path.join(DIAGRAMS_DIR, "flow_grok3_mini.png"))
    generate_ministral_flow(os.path.join(DIAGRAMS_DIR, "flow_ministral_14b.png"))
    generate_deepseek_flow(os.path.join(DIAGRAMS_DIR, "flow_deepseek_r1_distill.png"))

    # Data-driven matplotlib charts
    summary = load_results(split)
    if summary:
        generate_accuracy_chart(summary, os.path.join(DIAGRAMS_DIR, "accuracy_comparison.png"))
        generate_response_time_chart(summary, os.path.join(DIAGRAMS_DIR, "response_time.png"))
        generate_combined_chart(summary, os.path.join(DIAGRAMS_DIR, "combined_metrics.png"))
        generate_error_rate_chart(summary, os.path.join(DIAGRAMS_DIR, "error_rate.png"))
        generate_radar_chart(summary, os.path.join(DIAGRAMS_DIR, "radar_comparison.png"))
        generate_per_model_time_histogram(split)
    else:
        print("  No results found yet \u2014 skipping data-driven charts.")

    print("\nDiagram generation complete.")


if __name__ == "__main__":
    generate_all_diagrams()
