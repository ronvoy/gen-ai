"""
Report & Slide Generation Script

Produces report/report.md and report/slide.md from evaluation results.
"""

import os
import json

from config import RESULTS_DIR, REPORT_DIR, MODELS, MODEL_INFO


def load_results(split="test"):
    path = os.path.join(RESULTS_DIR, f"summary_{split}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _short(model_id):
    return MODEL_INFO.get(model_id, {}).get("name", model_id.split("/")[-1])


# -----------------------------------------------------------------------
# report.md
# -----------------------------------------------------------------------

def generate_report(split="test"):
    os.makedirs(REPORT_DIR, exist_ok=True)
    summary = load_results(split)

    md = []
    md.append("# LAMBADA Benchmark Evaluation Report\n")

    # --- Section 1 ---
    md.append("## 1. Introduction\n")
    md.append(
        "This report presents the evaluation of three Small Language Models (SLMs) on the "
        "**LAMBADA** (LAnguage Modeling Broadened to Account for Discourse Aspects) benchmark. "
        "The LAMBADA dataset tests a model's ability to predict the last word of a passage, "
        "requiring understanding of long-range dependencies in text.\n"
    )

    md.append("### Models Evaluated\n")
    md.append("| Model | Developer | Parameters |")
    md.append("|-------|-----------|------------|")
    for m in MODELS:
        info = MODEL_INFO.get(m, {})
        md.append(
            f"| {info.get('name', m)} | {info.get('developer', 'N/A')} "
            f"| {info.get('params', 'N/A')} |"
        )

    md.append("\n### Evaluation Setup\n")
    md.append("- **API**: OpenRouter (chat completions endpoint)")
    md.append("- **Temperature**: 0.0 (deterministic)")
    md.append("- **Task**: Last-word prediction (zero-shot)")
    md.append("- **Metric**: Exact match accuracy (case-insensitive, punctuation-stripped)\n")

    # --- Section 2 ---
    md.append("## 2. LAMBADA Dataset\n")
    md.append(
        "The LAMBADA dataset evaluates computational models for text understanding through "
        "word prediction. Key characteristics:\n"
    )
    md.append("- **Source**: Extracted from BookCorpus novels")
    md.append("- **Test Set**: 5,153 passages")
    md.append("- **Development Set**: 4,869 passages")
    md.append("- **Training Data**: 2,662 novels (~203 M words)")
    md.append("- **Task**: Predict the last word given the full passage context")
    md.append(
        "- **Key Property**: Humans can predict the word with full context "
        "but fail when given only the last sentence\n"
    )

    # --- Section 3 ---
    md.append("## 3. Methodology\n")
    md.append("```mermaid")
    md.append("graph TD")
    md.append("    A[Load LAMBADA Passages] --> B[Extract Context + Target Word]")
    md.append("    B --> C[Construct Prediction Prompt]")
    md.append("    C --> D[Query Model via OpenRouter API]")
    md.append("    D --> E[Parse & Normalize Response]")
    md.append("    E --> F[Compare with Target Word]")
    md.append("    F --> G[Calculate Accuracy Metrics]")
    md.append("    G --> H[Generate Report & Visualizations]")
    md.append("```\n")
    md.append("### Evaluation Pipeline\n")
    md.append("1. **Data Loading** \u2014 Sample passages from the LAMBADA test/dev splits")
    md.append(
        "2. **Context Extraction** \u2014 Split each passage into context "
        "(all words except last) and target (last word)"
    )
    md.append('3. **Prompt Construction** \u2014 "Complete with the next single word"')
    md.append("4. **API Query** \u2014 Send prompt to each model via OpenRouter (temperature=0.0)")
    md.append(
        "5. **Response Processing** \u2014 Extract first word, normalize "
        "(lowercase, strip punctuation)"
    )
    md.append("6. **Metric Calculation** \u2014 Exact match comparison between prediction and target\n")

    # --- Section 4 ---
    md.append("## 4. Results\n")
    if summary:
        md.append(
            f"### Performance on {split.title()} Split "
            f"({summary['num_samples']} samples)\n"
        )
        md.append("| Model | Accuracy | Correct | Total | Avg Response Time | Errors |")
        md.append("|-------|----------|---------|-------|-------------------|--------|")
        for m in summary["models"]:
            name = _short(m["model"])
            md.append(
                f"| {name} | {m['accuracy']*100:.1f}% | {m['correct']} | "
                f"{m['total']} | {m['avg_response_time']:.3f}s | {m['errors']} |"
            )

        best = max(summary["models"], key=lambda x: x["accuracy"])
        fastest = min(summary["models"], key=lambda x: x["avg_response_time"])

        md.append("\n### Key Findings\n")
        md.append(
            f"- **Best Accuracy**: {_short(best['model'])} "
            f"achieved **{best['accuracy']*100:.1f}%** accuracy"
        )
        md.append(
            f"- **Fastest Model**: {_short(fastest['model'])} "
            f"with **{fastest['avg_response_time']:.3f}s** average response time"
        )
        md.append("\n### Accuracy Comparison\n")
        md.append("![Accuracy Comparison](../diagrams/accuracy_comparison.png)\n")
        md.append("### Response Time Comparison\n")
        md.append("![Response Time](../diagrams/response_time.png)\n")
        md.append("### Combined Metrics\n")
        md.append("![Combined Metrics](../diagrams/combined_metrics.png)\n")
    else:
        md.append("*No evaluation results found. Run `evaluate_lambada.py` first.*\n")

    # --- Section 5 ---
    md.append("## 5. Model Descriptions\n")
    for model_id in MODELS:
        info = MODEL_INFO.get(model_id, {})
        md.append(f"### {info.get('name', model_id)}\n")
        md.append(f"- **Developer**: {info.get('developer', 'N/A')}")
        md.append(f"- **Parameters**: {info.get('params', 'N/A')}")
        md.append(f"- **Architecture**: {info.get('architecture', 'N/A')}")
        md.append(f"- **Description**: {info.get('description', 'N/A')}")
        md.append(f"- **Strengths**: {info.get('strengths', 'N/A')}")
        md.append(f"- **Weaknesses**: {info.get('weaknesses', 'N/A')}\n")

    # --- Section 6 ---
    md.append("## 6. Process Flow Diagrams\n")
    md.append("### Project Workflow\n")
    md.append("```mermaid")
    md.append("graph TD")
    md.append("    A[Start: Configuration] --> B[Load LAMBADA Dataset]")
    md.append("    B --> C[Sample Test Passages]")
    md.append("    C --> D{For Each Model}")
    md.append("    D --> E1[Grok-3-Mini]")
    md.append("    D --> E2[Ministral-14B]")
    md.append("    D --> E3[DeepSeek-R1-Distill-32B]")
    md.append("    E1 --> F1[OpenRouter API]")
    md.append("    E2 --> F2[OpenRouter API]")
    md.append("    E3 --> F3[OpenRouter API]")
    md.append("    F1 --> G[Collect & Compare Predictions]")
    md.append("    F2 --> G")
    md.append("    F3 --> G")
    md.append("    G --> H[Calculate Metrics]")
    md.append("    H --> I[Generate Report & Diagrams]")
    md.append("    I --> J[Streamlit Dashboard]")
    md.append("```\n")
    md.append("### Architecture Overview\n")
    md.append("```mermaid")
    md.append("graph LR")
    md.append("    subgraph Input")
    md.append("        A[LAMBADA Passage] --> B[Remove Last Word]")
    md.append("        B --> C[Context Prompt]")
    md.append("    end")
    md.append("    subgraph OpenRouter_API")
    md.append("        C --> D[Grok-3-Mini]")
    md.append("        C --> E[Ministral-14B]")
    md.append("        C --> F[DeepSeek-R1-Distill]")
    md.append("    end")
    md.append("    subgraph Evaluation")
    md.append("        D --> G[Predictions]")
    md.append("        E --> G")
    md.append("        F --> G")
    md.append("        G --> H{Exact Match?}")
    md.append("        H --> I[Accuracy Score]")
    md.append("    end")
    md.append("```\n")

    # --- Section 7 ---
    md.append("## 7. Conclusion\n")
    md.append(
        "This evaluation demonstrates the varying capabilities of three SLMs on the LAMBADA "
        "benchmark, which specifically tests long-range contextual understanding. The task "
        "requires models to leverage broad discourse context rather than relying solely on "
        "local patterns, making it a challenging test of language understanding.\n"
    )
    md.append(
        "The results highlight trade-offs between model size, accuracy, and inference speed, "
        "providing insights for selecting appropriate models based on deployment requirements.\n"
    )
    md.append("---\n")
    md.append("*Report generated automatically by the LAMBADA evaluation pipeline.*\n")

    report_path = os.path.join(REPORT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Report saved to {report_path}")
    return report_path


# -----------------------------------------------------------------------
# slide.md
# -----------------------------------------------------------------------

def generate_slides(split="test"):
    os.makedirs(REPORT_DIR, exist_ok=True)
    summary = load_results(split)

    md = []
    md.append("---")
    md.append("title: LAMBADA Benchmark Evaluation")
    md.append("subtitle: Evaluating Small Language Models on Long-Range Dependency Tasks")
    md.append("---\n")

    # Slide 1
    md.append("# Slide 1: Project Overview\n")
    md.append("## LAMBADA Benchmark Evaluation\n")
    md.append(
        "**Objective**: Evaluate three Small Language Models on the LAMBADA benchmark "
        "to assess their ability to understand long-range contextual dependencies.\n"
    )
    md.append("**Models Under Evaluation**:\n")
    md.append("- **Grok-3-Mini** (xAI) \u2014 Compact reasoning model with MoE architecture")
    md.append("- **Ministral-14B** (Mistral AI) \u2014 Efficient 14B parameter model with SWA")
    md.append(
        "- **DeepSeek-R1-Distill-Qwen-32B** (DeepSeek) \u2014 "
        "Distilled reasoning model based on Qwen\n"
    )
    md.append("---\n")

    # Slide 2
    md.append("# Slide 2: What is LAMBADA?\n")
    md.append("## The LAMBADA Benchmark\n")
    md.append("- **Full Name**: LAnguage Modeling Broadened to Account for Discourse Aspects")
    md.append("- **Task**: Predict the last word of a narrative passage")
    md.append("- **Key Challenge**: Requires understanding of broad discourse context")
    md.append(
        "- **Dataset**: 5,153 test passages + 4,869 development passages from BookCorpus"
    )
    md.append(
        "- **Key Property**: Humans succeed with full context but fail with the last sentence alone\n"
    )
    md.append("```mermaid")
    md.append("graph LR")
    md.append("    A[Full Passage Context] --> B[Model reads context]")
    md.append("    B --> C[Predicts last word]")
    md.append("    C --> D{Correct?}")
    md.append("```\n")
    md.append("---\n")

    # Slide 3
    md.append("# Slide 3: Evaluation Methodology\n")
    md.append("## How We Evaluate\n")
    md.append("1. **Load** passages from LAMBADA test set")
    md.append(
        "2. **Extract** context (everything except last word) and target (last word)"
    )
    md.append('3. **Prompt** each model: "Complete with the next single word"')
    md.append("4. **Compare** prediction vs. target (case-insensitive, normalized)")
    md.append("5. **Calculate** accuracy, response time, error rate\n")
    md.append("**API**: OpenRouter (unified endpoint for all models)  ")
    md.append("**Temperature**: 0.0 (deterministic output)\n")
    md.append("---\n")

    # Slide 4
    md.append("# Slide 4: Model Architectures\n")
    md.append("## Comparing the Three SLMs\n")
    md.append("| Feature | Grok-3-Mini | Ministral-14B | DeepSeek-R1-Distill |")
    md.append("|---------|-------------|---------------|---------------------|")
    md.append("| Developer | xAI | Mistral AI | DeepSeek |")
    md.append("| Parameters | ~3B (est.) | 14B | 32B |")
    md.append(
        "| Architecture | Decoder + MoE | Decoder + SWA | Qwen + Distilled Reasoning |"
    )
    md.append("| Strength | Speed + Reasoning | Multilingual | Complex Reasoning |\n")
    md.append("---\n")

    # Slide 5
    md.append("# Slide 5: Results\n")
    md.append("## Performance Metrics\n")
    if summary:
        md.append("| Model | Accuracy | Avg Response Time | Errors |")
        md.append("|-------|----------|-------------------|--------|")
        for m in summary["models"]:
            name = _short(m["model"])
            md.append(
                f"| {name} | {m['accuracy']*100:.1f}% "
                f"| {m['avg_response_time']:.3f}s | {m['errors']} |"
            )
        best = max(summary["models"], key=lambda x: x["accuracy"])
        md.append(
            f"\n**Winner**: {_short(best['model'])} "
            f"with {best['accuracy']*100:.1f}% accuracy\n"
        )
    else:
        md.append("*Results pending evaluation run.*\n")
    md.append("---\n")

    # Slide 6
    md.append("# Slide 6: Workflow Diagram\n")
    md.append("## Project Pipeline\n")
    md.append("```mermaid")
    md.append("graph TD")
    md.append("    A[Configuration] --> B[Load Dataset]")
    md.append("    B --> C[Sample Passages]")
    md.append("    C --> D[Evaluate Models]")
    md.append("    D --> E[Calculate Metrics]")
    md.append("    E --> F[Generate Diagrams]")
    md.append("    F --> G[Generate Report]")
    md.append("    G --> H[Streamlit Dashboard]")
    md.append("```\n")
    md.append("---\n")

    # Slide 7
    md.append("# Slide 7: Key Takeaways\n")
    md.append("## Conclusions\n")
    md.append("- LAMBADA effectively tests long-range dependency understanding")
    md.append("- Model size influences accuracy but the relationship is not linear")
    md.append("- API-based evaluation enables fair comparison without local GPU requirements")
    md.append("- Trade-offs exist between accuracy, speed, and cost")
    md.append("- All three models exhibit distinct strengths suited to different use cases\n")
    md.append("---\n")

    # Slide 8
    md.append("# Slide 8: Interactive Dashboard\n")
    md.append("## Streamlit Application Features\n")
    md.append("- **Model Selection**: Dropdown to choose and compare LLMs")
    md.append("- **Dataset Explorer**: Browse LAMBADA test, dev, and rejected data")
    md.append("- **Metrics Display**: Accuracy, response time, per-sample results")
    md.append("- **Visualizations**: Bar charts, comparison tables, mermaid diagrams")
    md.append(
        "- **Working Principles**: Detailed explanation of each model's architecture\n"
    )
    md.append("---\n")
    md.append("*Generated by LAMBADA Evaluation Pipeline*\n")

    slides_path = os.path.join(REPORT_DIR, "slide.md")
    with open(slides_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Slides saved to {slides_path}")
    return slides_path


# -----------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------

def generate_all_reports(split="test"):
    print("\nGenerating reports...")
    print("=" * 40)
    generate_report(split)
    generate_slides(split)
    print("\nReport generation complete.")


if __name__ == "__main__":
    generate_all_reports()
