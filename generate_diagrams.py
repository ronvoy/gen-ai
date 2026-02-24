"""
Diagram Generation Script

Produces matplotlib comparison charts and mermaid workflow diagrams
saved as PNG files.
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


def _model_short(model_id):
    return MODEL_INFO.get(model_id, {}).get("name", model_id.split("/")[-1])


def _model_colors():
    return [MODEL_INFO[m]["color"] for m in MODELS if m in MODEL_INFO]


# ---------------------------------------------------------------------------
# Matplotlib charts
# ---------------------------------------------------------------------------

def generate_accuracy_chart(summary, output_path):
    models = [_model_short(m["model"]) for m in summary["models"]]
    accs = [m["accuracy"] * 100 for m in summary["models"]]
    colors = _model_colors()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accs, color=colors[: len(models)], width=0.5, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"LAMBADA Benchmark \u2014 Accuracy Comparison ({summary['split']} split)",
        fontsize=14,
    )
    ax.set_ylim(0, max(accs) * 1.25 if accs else 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_response_time_chart(summary, output_path):
    models = [_model_short(m["model"]) for m in summary["models"]]
    times = [m["avg_response_time"] for m in summary["models"]]
    colors = _model_colors()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, times, color=colors[: len(models)], width=0.5, edgecolor="white")
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{t:.3f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )
    ax.set_ylabel("Avg Response Time (s)", fontsize=12)
    ax.set_title(
        f"LAMBADA Benchmark \u2014 Response Time ({summary['split']} split)", fontsize=14
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_combined_chart(summary, output_path):
    models = [_model_short(m["model"]) for m in summary["models"]]
    accs = [m["accuracy"] * 100 for m in summary["models"]]
    times = [m["avg_response_time"] for m in summary["models"]]
    colors = _model_colors()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars1 = ax1.bar(models, accs, color=colors[: len(models)], width=0.5)
    for bar, acc in zip(bars1, accs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy")
    ax1.set_ylim(0, max(accs) * 1.25 if accs else 100)
    ax1.grid(axis="y", alpha=0.3)

    bars2 = ax2.bar(models, times, color=colors[: len(models)], width=0.5)
    for bar, t in zip(bars2, times):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{t:.3f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax2.set_ylabel("Avg Response Time (s)")
    ax2.set_title("Response Time")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"LAMBADA Benchmark Comparison ({summary['split']} split)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Mermaid -> PNG  (via mermaid.ink public renderer)
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
        "    subgraph OpenRouter API\n"
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
        "    A[\"Full Passage<br/>'she looked at the sky and said it was'\"]\n"
        "    A --> B[\"Context<br/>'she looked at the sky and said it was'\"]\n"
        "    A --> C[\"Target Word<br/>'beautiful'\"]\n"
        "    B --> D[Send to LLM]\n"
        "    D --> E[\"Prediction<br/>'beautiful'\"]\n"
        "    E --> F{Match?}\n"
        "    C --> F\n"
        "    F -->|Yes| G[Correct]\n"
        "    F -->|No| H[Incorrect]\n"
        "    style G fill:#c8e6c9\n"
        "    style H fill:#ffcdd2"
    )
    return save_mermaid_as_png(mermaid, output_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_diagrams(split="test"):
    os.makedirs(DIAGRAMS_DIR, exist_ok=True)

    print("\nGenerating diagrams...")
    print("=" * 40)

    generate_workflow_diagram(os.path.join(DIAGRAMS_DIR, "project_workflow.png"))
    generate_architecture_diagram(os.path.join(DIAGRAMS_DIR, "architecture_comparison.png"))
    generate_evaluation_pipeline_diagram(os.path.join(DIAGRAMS_DIR, "evaluation_pipeline.png"))
    generate_lambada_task_diagram(os.path.join(DIAGRAMS_DIR, "lambada_task.png"))

    summary = load_results(split)
    if summary:
        generate_accuracy_chart(summary, os.path.join(DIAGRAMS_DIR, "accuracy_comparison.png"))
        generate_response_time_chart(summary, os.path.join(DIAGRAMS_DIR, "response_time.png"))
        generate_combined_chart(summary, os.path.join(DIAGRAMS_DIR, "combined_metrics.png"))
    else:
        print("  No results found yet \u2014 skipping data-driven charts.")

    print("\nDiagram generation complete.")


if __name__ == "__main__":
    generate_all_diagrams()
