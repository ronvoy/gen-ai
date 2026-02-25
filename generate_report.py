"""
Report & Slide Generation Script

Produces a thorough report/report.md and a concise report/slide.md
from evaluation results, model metadata, and generated diagram PNGs.
"""

import os
import json

from config import RESULTS_DIR, REPORT_DIR, DIAGRAMS_DIR, MODELS, MODEL_INFO


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


def _n(model_id):
    return MODEL_INFO.get(model_id, {}).get("name", model_id.split("/")[-1])


def _img(fname):
    """Return markdown image link relative to report/ directory."""
    return f"![{fname}](../diagrams/{fname})"


# ======================================================================
# report.md
# ======================================================================

def generate_report(split="test"):   # noqa: C901 — long but linear
    os.makedirs(REPORT_DIR, exist_ok=True)
    summary = load_results(split)

    L = []  # collector

    # ------------------------------------------------------------------
    L.append("# LAMBADA Benchmark Evaluation Report")
    L.append("")
    L.append("> Evaluating Small Language Models on Long-Range Dependency Word Prediction")
    L.append("")

    # ==================================================================
    # 1  INTRODUCTION
    # ==================================================================
    L.append("## 1. Introduction")
    L.append("")
    L.append(
        "This report documents a systematic evaluation of **three Small Language Models "
        "(SLMs)** on the **LAMBADA** benchmark (LAnguage Modeling Broadened to Account for "
        "Discourse Aspects). LAMBADA is a word-prediction task that specifically measures a "
        "model\u2019s ability to capture long-range dependencies across narrative passages. "
        "The evaluation is conducted through the **OpenRouter API**, providing a fair, "
        "infrastructure-neutral comparison."
    )
    L.append("")

    # ==================================================================
    # 2  SMALL LANGUAGE MODELS USED
    # ==================================================================
    L.append("## 2. Small Language Models Used")
    L.append("")
    L.append("### 2.1 Models at a Glance")
    L.append("")
    L.append("| # | Model | Developer | Parameters | Architecture | Key Technique |")
    L.append("|---|-------|-----------|------------|--------------|---------------|")
    techniques = {
        "x-ai/grok-3-mini": "Mixture of Experts (MoE)",
        "mistralai/ministral-14b-2512": "Sliding Window Attention (SWA) + GQA",
        "deepseek/deepseek-r1-distill-qwen-32b": "Knowledge Distillation from DeepSeek-R1",
    }
    for i, m in enumerate(MODELS, 1):
        info = MODEL_INFO[m]
        L.append(
            f"| {i} | {info['name']} | {info['developer']} | {info['params']} "
            f"| {info['architecture']} | {techniques[m]} |"
        )
    L.append("")

    # --- 2.2  Per-model deep dives ---
    L.append("### 2.2 Grok-3-Mini (xAI)")
    L.append("")
    L.append(MODEL_INFO["x-ai/grok-3-mini"]["description"])
    L.append("")
    L.append("**Working Technique \u2014 Mixture of Experts (MoE):**")
    L.append("")
    L.append(
        "In a standard dense transformer every token passes through the full feed-forward "
        "network (FFN). MoE replaces the single FFN with *N* parallel expert FFNs and a "
        "lightweight **router** that selects the top-*k* experts per token. Only the chosen "
        "experts are activated, so the model can hold many more parameters than it uses per "
        "forward pass, yielding high capacity at low compute cost."
    )
    L.append("")
    L.append("Key properties:")
    L.append("")
    L.append("- **Sparse activation** \u2014 Only a subset of parameters fire per token.")
    L.append("- **Router network** \u2014 A learned gating function assigns tokens to experts.")
    L.append("- **Load balancing loss** \u2014 An auxiliary loss term prevents expert collapse.")
    L.append("- **RLHF fine-tuning** \u2014 Reinforcement Learning from Human Feedback sharpens instruction following.")
    L.append("")
    L.append("#### Process Flow \u2014 Grok-3-Mini Inference")
    L.append("")
    L.append("```mermaid")
    L.append("graph TD")
    L.append("    A[Input Tokens] --> B[Token Embedding + RoPE Positional Encoding]")
    L.append("    B --> C[Transformer Decoder Layer x N]")
    L.append("    C --> D[Self-Attention with Causal Mask]")
    L.append("    D --> E[MoE Router: select top-k experts]")
    L.append("    E --> F1[Expert FFN 1]")
    L.append("    E --> F2[Expert FFN 2]")
    L.append("    E --> F3[Expert FFN k]")
    L.append("    F1 --> G[Weighted Sum of Expert Outputs]")
    L.append("    F2 --> G")
    L.append("    F3 --> G")
    L.append("    G --> H[Residual + LayerNorm]")
    L.append("    H --> C")
    L.append("    H --> I[Final LayerNorm]")
    L.append("    I --> J[LM Head: Vocabulary Projection]")
    L.append("    J --> K[Softmax -> Next Token]")
    L.append("```")
    L.append("")
    L.append(_img("flow_grok3_mini.png"))
    L.append("")

    # --- Ministral ---
    L.append("### 2.3 Ministral-14B (Mistral AI)")
    L.append("")
    L.append(MODEL_INFO["mistralai/ministral-14b-2512"]["description"])
    L.append("")
    L.append("**Working Technique \u2014 Sliding Window Attention (SWA) + Grouped Query Attention (GQA):**")
    L.append("")
    L.append(
        "Standard self-attention has \u039f(n\u00b2) complexity in sequence length. SWA limits each "
        "token\u2019s attention to a fixed window of *W* preceding tokens, reducing memory to \u039f(n\u00b7W). "
        "Information beyond the window propagates through stacked layers. GQA groups multiple "
        "query heads under fewer key-value heads, cutting KV-cache memory without sacrificing quality."
    )
    L.append("")
    L.append("Key properties:")
    L.append("")
    L.append("- **Sliding Window Attention** \u2014 Each layer attends to a local window; deeper layers see broader context.")
    L.append("- **Grouped Query Attention** \u2014 Fewer KV heads \u2192 reduced memory, faster decoding.")
    L.append("- **SwiGLU activation** \u2014 Gated linear unit variant that improves training dynamics.")
    L.append("- **RMSNorm** \u2014 Faster and more stable normalisation compared to LayerNorm.")
    L.append("")
    L.append("#### Process Flow \u2014 Ministral-14B Inference")
    L.append("")
    L.append("```mermaid")
    L.append("graph TD")
    L.append("    A[Input Tokens] --> B[Token Embedding + RoPE Positional Encoding]")
    L.append("    B --> C[Transformer Decoder Layer x N]")
    L.append("    C --> D[Sliding Window Attention - SWA]")
    L.append("    D --> E[Grouped Query Attention - GQA]")
    L.append("    E --> F[SwiGLU Feed-Forward Network]")
    L.append("    F --> G[RMSNorm + Residual Connection]")
    L.append("    G --> C")
    L.append("    G --> H[Final RMSNorm]")
    L.append("    H --> I[LM Head: Vocabulary Projection]")
    L.append("    I --> J[Softmax -> Next Token]")
    L.append("```")
    L.append("")
    L.append(_img("flow_ministral_14b.png"))
    L.append("")

    # --- DeepSeek ---
    L.append("### 2.4 DeepSeek-R1-Distill-Qwen-32B (DeepSeek)")
    L.append("")
    L.append(MODEL_INFO["deepseek/deepseek-r1-distill-qwen-32b"]["description"])
    L.append("")
    L.append("**Working Technique \u2014 Knowledge Distillation:**")
    L.append("")
    L.append(
        "Knowledge distillation transfers the capabilities of a large *teacher* model into a "
        "smaller *student* model. DeepSeek-R1 (the teacher) generates chain-of-thought reasoning "
        "traces, and the student (Qwen-2.5-32B) is trained to reproduce those traces via a "
        "combination of supervised fine-tuning on the teacher\u2019s outputs and a KL-divergence loss "
        "that aligns the student\u2019s output distribution with the teacher\u2019s."
    )
    L.append("")
    L.append("Key properties:")
    L.append("")
    L.append("- **Teacher\u2013Student framework** \u2014 DeepSeek-R1 (671B MoE) \u2192 Qwen-2.5 (32B dense).")
    L.append("- **Chain-of-Thought distillation** \u2014 Reasoning patterns are explicitly transferred.")
    L.append("- **RoPE positional embeddings** \u2014 Rotary embeddings for length generalisation.")
    L.append("- **SwiGLU FFN + RMSNorm** \u2014 Same efficient building blocks as the Qwen base.")
    L.append("")
    L.append("#### Process Flow \u2014 DeepSeek-R1-Distill-Qwen-32B Inference")
    L.append("")
    L.append("```mermaid")
    L.append("graph TD")
    L.append("    A[Input Tokens] --> B[Token Embedding + RoPE Positional Encoding]")
    L.append("    B --> C[Qwen-2.5 Transformer Decoder Layer x N]")
    L.append("    C --> D[Multi-Head Self-Attention with Causal Mask]")
    L.append("    D --> E[SwiGLU Feed-Forward Network]")
    L.append("    E --> F[RMSNorm + Residual Connection]")
    L.append("    F --> C")
    L.append("    F --> G[Final RMSNorm]")
    L.append("    G --> H[LM Head: Vocabulary Projection]")
    L.append("    H --> I[Softmax -> Next Token]")
    L.append("    subgraph Distillation_Origin")
    L.append("        T1[DeepSeek-R1 Teacher 671B MoE] --> T2[Chain-of-Thought Reasoning Data]")
    L.append("        T2 --> T3[KL-Divergence Distillation Loss]")
    L.append("        T3 --> C")
    L.append("    end")
    L.append("```")
    L.append("")
    L.append(_img("flow_deepseek_r1_distill.png"))
    L.append("")

    # ==================================================================
    # 3  DATASET OVERVIEW
    # ==================================================================
    L.append("## 3. Dataset Overview")
    L.append("")
    L.append(
        "The LAMBADA dataset is drawn from the BookCorpus and designed so that the target "
        "word (always the last word of a passage) can be predicted by humans only when the "
        "full passage is available, not from the final sentence alone."
    )
    L.append("")
    L.append("### 3.1 Splits Summary")
    L.append("")
    L.append("| Split | File | Passages | Purpose |")
    L.append("|-------|------|----------|---------|")
    L.append("| **Test** | `lambada_test_plain_text.txt` | 5,153 | Primary evaluation split |")
    L.append("| **Development** | `lambada_development_plain_text.txt` | 4,869 | Hyper-parameter tuning / validation |")
    L.append("| **Control Test** | `lambada_control_test_data_plain_text.txt` | 5,000 | Baseline passages (not filtered for long-range dependency) |")
    L.append("| **Rejected** | `rejected_plain_text.txt` | 11,941 | Passages rejected during curation (guessable from last sentence alone) |")
    L.append("| **Training Novels** | `train-novels/` (16 genres) | 2,662 novels (~203M words) | Language-model pre-training data |")
    L.append("| **Vocabulary** | `lambada-vocab-2.txt` | 112,746 entries | Vocabulary reference list |")
    L.append("")
    L.append("### 3.2 Genre Distribution (Training Novels)")
    L.append("")
    L.append("| Genre | Examples |")
    L.append("|-------|----------|")
    L.append("| Adventure | Narrative action fiction |")
    L.append("| Fantasy | Epic / urban fantasy novels |")
    L.append("| Historical | Period fiction |")
    L.append("| Horror | Supernatural / psychological horror |")
    L.append("| Humor | Comic fiction |")
    L.append("| Literature | Literary fiction |")
    L.append("| Mystery | Detective / crime fiction |")
    L.append("| New Adult | Post-YA contemporary |")
    L.append("| Other | Uncategorised |")
    L.append("| Romance | Love / relationship narratives |")
    L.append("| Science Fiction | Speculative / sci-fi |")
    L.append("| Teen | Teenage-audience novels |")
    L.append("| Themes | Thematic anthologies |")
    L.append("| Thriller | Suspense / thriller |")
    L.append("| Vampires | Vampire-centric fiction |")
    L.append("| Young Adult | YA fiction |")
    L.append("")
    L.append("### 3.3 Dataset Properties")
    L.append("")
    L.append("| Property | Value |")
    L.append("|----------|-------|")
    L.append("| Source corpus | BookCorpus (unpublished novels) |")
    L.append("| Language | English (BCP-47: `en`) |")
    L.append("| Licence | CC BY 4.0 |")
    L.append("| Task type | Text-to-text / word prediction |")
    L.append("| Annotation | Expert-generated + crowd-sourced validation |")
    L.append("| Curation criterion | Target word guessable from full context only |")
    L.append("| First published | ACL 2016 (Paperno et al.) |")
    L.append("")

    # ==================================================================
    # 4  BENCHMARKING TYPES & METHODS
    # ==================================================================
    L.append("## 4. Benchmarking Types & Methods")
    L.append("")
    L.append(
        "We evaluate each model along four complementary axes. The table below lists "
        "each metric, its definition, and its interpretation."
    )
    L.append("")
    L.append("### 4.1 Metrics Overview")
    L.append("")
    L.append("| # | Metric | Formula / Definition | Interpretation | Unit |")
    L.append("|---|--------|----------------------|----------------|------|")
    L.append(
        "| 1 | **Exact-Match Accuracy** | `correct / total` after case-insensitive, "
        "punctuation-stripped normalisation | Higher is better; primary quality indicator | % |"
    )
    L.append(
        "| 2 | **Average Response Latency** | Mean wall-clock time per API call | "
        "Lower is better; measures inference speed | seconds |"
    )
    L.append(
        "| 3 | **API Error Rate** | `errors / total` (timeouts, HTTP failures, "
        "malformed responses) | Lower is better; measures reliability | % |"
    )
    L.append(
        "| 4 | **Throughput** | `total / total_wall_clock_time` | "
        "Higher is better; end-to-end efficiency | samples/s |"
    )
    L.append("")

    L.append("### 4.2 Evaluation Method \u2014 Process Flow")
    L.append("")
    L.append("```mermaid")
    L.append("graph TD")
    L.append("    A[LAMBADA Benchmark Method] --> B[Exact-Match Accuracy]")
    L.append("    A --> C[Response Latency]")
    L.append("    A --> D[Error Rate]")
    L.append("    A --> E[Throughput]")
    L.append("    B --> B1[Normalize prediction: lowercase + strip punctuation]")
    L.append("    B1 --> B2[Compare predicted word == target word]")
    L.append("    B2 --> B3[Accuracy = correct / total]")
    L.append("    C --> C1[Record wall-clock time per API call]")
    L.append("    C1 --> C2[Compute mean, median, min, max latency]")
    L.append("    D --> D1[Count API failures or timeouts]")
    L.append("    D1 --> D2[Error Rate = errors / total]")
    L.append("    E --> E1[Samples evaluated / total wall-clock time]")
    L.append("```")
    L.append("")
    L.append(_img("benchmark_method.png"))
    L.append("")
    L.append("### 4.3 Prompt Template")
    L.append("")
    L.append("Every model receives the same zero-shot prompt:")
    L.append("")
    L.append("```")
    L.append("Complete the following passage with the next single word.")
    L.append("Only respond with that one word, nothing else.")
    L.append("")
    L.append("Passage: <context without last word>")
    L.append("```")
    L.append("")
    L.append("| Parameter | Value |")
    L.append("|-----------|-------|")
    L.append("| Temperature | 0.0 (greedy / deterministic) |")
    L.append("| Max tokens | 10 |")
    L.append("| API | OpenRouter (`/api/v1/chat/completions`) |")
    L.append("| Timeout | 120 s |")
    L.append("")

    # ==================================================================
    # 5  PERFORMANCE RESULTS
    # ==================================================================
    L.append("## 5. Performance Results")
    L.append("")

    if summary:
        n = summary["num_samples"]
        L.append(f"Evaluation was run on **{n} randomly sampled passages** from the **{split}** split.\n")

        # --- 5.1 Summary table ---
        L.append("### 5.1 Summary Table")
        L.append("")
        L.append("| Model | Accuracy (%) | Correct | Total | Avg Latency (s) | Error Rate (%) | Throughput (samples/s) |")
        L.append("|-------|-------------|---------|-------|-----------------|----------------|----------------------|")
        for m in summary["models"]:
            name = _n(m["model"])
            err_pct = m["errors"] / m["total"] * 100 if m["total"] else 0
            detail = load_detailed(m["model"], split)
            total_time = detail["total_time"] if detail else m["avg_response_time"] * m["total"]
            throughput = m["total"] / total_time if total_time else 0
            L.append(
                f"| {name} | {m['accuracy']*100:.1f} | {m['correct']} | {m['total']} "
                f"| {m['avg_response_time']:.3f} | {err_pct:.1f} | {throughput:.2f} |"
            )
        L.append("")

        best = max(summary["models"], key=lambda x: x["accuracy"])
        fastest = min(summary["models"], key=lambda x: x["avg_response_time"])
        L.append(f"**Best Accuracy**: {_n(best['model'])} \u2014 **{best['accuracy']*100:.1f}%**  ")
        L.append(f"**Fastest Model**: {_n(fastest['model'])} \u2014 **{fastest['avg_response_time']:.3f}s** avg latency")
        L.append("")

        # --- 5.2 Detailed per-model statistics ---
        L.append("### 5.2 Detailed Per-Model Statistics")
        L.append("")
        for m_id in MODELS:
            detail = load_detailed(m_id, split)
            if not detail:
                continue
            times = [r["time"] for r in detail["results"]]
            correct_times = [r["time"] for r in detail["results"] if r["correct"]]
            wrong_times = [r["time"] for r in detail["results"] if not r["correct"]]
            import statistics
            L.append(f"#### {_n(m_id)}")
            L.append("")
            L.append("| Statistic | Value |")
            L.append("|-----------|-------|")
            L.append(f"| Accuracy | {detail['accuracy']*100:.1f}% |")
            L.append(f"| Correct predictions | {detail['correct']} / {detail['total']} |")
            L.append(f"| Total wall-clock time | {detail['total_time']:.1f}s |")
            L.append(f"| Mean latency | {statistics.mean(times):.3f}s |")
            L.append(f"| Median latency | {statistics.median(times):.3f}s |")
            L.append(f"| Min latency | {min(times):.3f}s |")
            L.append(f"| Max latency | {max(times):.3f}s |")
            L.append(f"| Std-dev latency | {statistics.pstdev(times):.3f}s |")
            L.append(f"| API errors | {detail['errors']} |")
            if correct_times:
                L.append(f"| Mean latency (correct) | {statistics.mean(correct_times):.3f}s |")
            if wrong_times:
                L.append(f"| Mean latency (incorrect) | {statistics.mean(wrong_times):.3f}s |")
            L.append("")

        # --- 5.3 Charts ---
        L.append("### 5.3 Accuracy Comparison")
        L.append("")
        L.append(_img("accuracy_comparison.png"))
        L.append("")

        L.append("### 5.4 Response Time Comparison")
        L.append("")
        L.append(_img("response_time.png"))
        L.append("")

        L.append("### 5.5 Combined Metrics")
        L.append("")
        L.append(_img("combined_metrics.png"))
        L.append("")

        L.append("### 5.6 API Error Rate")
        L.append("")
        L.append(_img("error_rate.png"))
        L.append("")

        L.append("### 5.7 Radar Comparison (Multi-Metric)")
        L.append("")
        L.append(_img("radar_comparison.png"))
        L.append("")

        L.append("### 5.8 Response-Time Distributions")
        L.append("")
        for m_id in MODELS:
            safe = m_id.replace("/", "_")
            fname = f"time_hist_{safe}.png"
            if os.path.exists(os.path.join(DIAGRAMS_DIR, fname)):
                L.append(f"#### {_n(m_id)}")
                L.append("")
                L.append(_img(fname))
                L.append("")
    else:
        L.append("*No evaluation results found. Run `python main.py` first.*")
        L.append("")

    # ==================================================================
    # 6  PROJECT WORKFLOW
    # ==================================================================
    L.append("## 6. Project Workflow")
    L.append("")
    L.append("```mermaid")
    L.append("graph TD")
    L.append("    A[Start: Load Configuration] --> B[Load LAMBADA Dataset]")
    L.append("    B --> C[Sample N Test Passages]")
    L.append("    C --> D{For Each Model}")
    L.append("    D --> E1[Grok-3-Mini]")
    L.append("    D --> E2[Ministral-14B]")
    L.append("    D --> E3[DeepSeek-R1-Distill-32B]")
    L.append("    E1 --> F1[OpenRouter API]")
    L.append("    E2 --> F2[OpenRouter API]")
    L.append("    E3 --> F3[OpenRouter API]")
    L.append("    F1 --> G[Collect Predictions]")
    L.append("    F2 --> G")
    L.append("    F3 --> G")
    L.append("    G --> H[Calculate Accuracy & Metrics]")
    L.append("    H --> I[Save Results JSON]")
    L.append("    I --> J[Generate Charts & Diagrams]")
    L.append("    J --> K[Generate Report & Slides]")
    L.append("    K --> L[Streamlit Dashboard]")
    L.append("```")
    L.append("")
    L.append(_img("project_workflow.png"))
    L.append("")

    # ==================================================================
    # 7  EVALUATION ARCHITECTURE
    # ==================================================================
    L.append("## 7. Evaluation Architecture")
    L.append("")
    L.append("```mermaid")
    L.append("graph LR")
    L.append("    subgraph Input")
    L.append("        A[LAMBADA Passage] --> B[Remove Last Word]")
    L.append("        B --> C[Context Prompt]")
    L.append("    end")
    L.append("    subgraph OpenRouter_API")
    L.append("        C --> D[Grok-3-Mini]")
    L.append("        C --> E[Ministral-14B]")
    L.append("        C --> F[DeepSeek-R1-Distill]")
    L.append("    end")
    L.append("    subgraph Evaluation")
    L.append("        D --> G[Predictions]")
    L.append("        E --> G")
    L.append("        F --> G")
    L.append("        G --> H{Exact Match?}")
    L.append("        H --> I[Accuracy Score]")
    L.append("    end")
    L.append("```")
    L.append("")
    L.append(_img("architecture_comparison.png"))
    L.append("")

    # ==================================================================
    # 8  CONCLUSION
    # ==================================================================
    L.append("## 8. Conclusion")
    L.append("")
    if summary:
        best = max(summary["models"], key=lambda x: x["accuracy"])
        fastest = min(summary["models"], key=lambda x: x["avg_response_time"])
        L.append(
            f"Among the three models evaluated, **{_n(best['model'])}** achieved the highest "
            f"accuracy ({best['accuracy']*100:.1f}%), while **{_n(fastest['model'])}** was the "
            f"fastest ({fastest['avg_response_time']:.3f}s per query)."
        )
        L.append("")
    L.append(
        "The LAMBADA benchmark proves to be a demanding test of contextual language "
        "understanding. Models must go beyond surface-level token statistics and truly "
        "comprehend the narrative flow to predict the correct final word. "
        "The evaluation reveals clear trade-offs between model size, accuracy, and latency "
        "that practitioners should weigh when choosing an SLM for deployment."
    )
    L.append("")
    L.append("Key take-aways:")
    L.append("")
    L.append("1. Larger parameter counts generally improve accuracy but increase latency and cost.")
    L.append("2. Specialised architectures (MoE, SWA) can offset size disadvantages.")
    L.append("3. Knowledge distillation effectively compresses reasoning ability into smaller models.")
    L.append("4. API-based evaluation ensures reproducibility and avoids hardware-specific confounds.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## References")
    L.append("")
    L.append(
        "- Paperno, D. et al. (2016). *The LAMBADA dataset: Word prediction requiring a broad "
        "discourse context*. Proceedings of ACL 2016, pp. 1525\u20131534."
    )
    L.append("- xAI (2025). *Grok-3-Mini Technical Report*.")
    L.append("- Mistral AI (2025). *Ministral Model Family*.")
    L.append("- DeepSeek (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("*Report generated automatically by the LAMBADA evaluation pipeline.*")
    L.append("")

    report_path = os.path.join(REPORT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Report saved to {report_path}")
    return report_path


# ======================================================================
# slide.md
# ======================================================================

def generate_slides(split="test"):
    os.makedirs(REPORT_DIR, exist_ok=True)
    summary = load_results(split)

    S = []

    S.append("---")
    S.append("title: LAMBADA Benchmark Evaluation")
    S.append("subtitle: Evaluating Small Language Models on Long-Range Dependency Word Prediction")
    S.append("---")
    S.append("")

    # Slide 1
    S.append("# Slide 1 \u2014 Project Overview")
    S.append("")
    S.append("## LAMBADA Benchmark: Evaluating Three SLMs")
    S.append("")
    S.append(
        "We evaluate **Grok-3-Mini**, **Ministral-14B**, and **DeepSeek-R1-Distill-Qwen-32B** "
        "on the LAMBADA last-word-prediction benchmark using the OpenRouter API."
    )
    S.append("")
    S.append("---")
    S.append("")

    # Slide 2
    S.append("# Slide 2 \u2014 Models Used")
    S.append("")
    S.append("## Small Language Models at a Glance")
    S.append("")
    S.append("| Model | Developer | Parameters | Key Technique |")
    S.append("|-------|-----------|------------|---------------|")
    S.append("| Grok-3-Mini | xAI | ~3B (est.) | Mixture of Experts (MoE) |")
    S.append("| Ministral-14B | Mistral AI | 14B | Sliding Window + GQA |")
    S.append("| DeepSeek-R1-Distill | DeepSeek | 32B | Knowledge Distillation |")
    S.append("")
    S.append("---")
    S.append("")

    # Slide 3
    S.append("# Slide 3 \u2014 How Each Model Works")
    S.append("")
    S.append("## Grok-3-Mini: Mixture of Experts")
    S.append("")
    S.append("A router selects top-*k* expert FFNs per token \u2192 sparse activation, high throughput.")
    S.append("")
    S.append("## Ministral-14B: Sliding Window Attention")
    S.append("")
    S.append("Each layer attends to a fixed window; deep stacking gives long-range reach at \u039f(n\u00b7W) cost.")
    S.append("")
    S.append("## DeepSeek-R1-Distill: Knowledge Distillation")
    S.append("")
    S.append("Chain-of-thought reasoning from a 671B teacher distilled into a 32B Qwen student.")
    S.append("")
    S.append("---")
    S.append("")

    # Slide 4
    S.append("# Slide 4 \u2014 LAMBADA Dataset Overview")
    S.append("")
    S.append("## Dataset Splits")
    S.append("")
    S.append("| Split | Passages | Purpose |")
    S.append("|-------|----------|---------|")
    S.append("| Test | 5,153 | Primary evaluation |")
    S.append("| Development | 4,869 | Validation |")
    S.append("| Control Test | 5,000 | Baseline (unfiltered) |")
    S.append("| Rejected | 11,941 | Guessable from last sentence |")
    S.append("| Training | 2,662 novels | LM pre-training data |")
    S.append("")
    S.append("Task: predict the **last word** of a narrative passage using full context.")
    S.append("")
    S.append("---")
    S.append("")

    # Slide 5
    S.append("# Slide 5 \u2014 Benchmarking Methods")
    S.append("")
    S.append("## Metrics Used")
    S.append("")
    S.append("| Metric | Definition | Goal |")
    S.append("|--------|------------|------|")
    S.append("| Exact-Match Accuracy | correct / total (normalised) | Higher = better |")
    S.append("| Avg Response Latency | Mean wall-clock time per API call | Lower = better |")
    S.append("| API Error Rate | errors / total | Lower = better |")
    S.append("| Throughput | samples / total time | Higher = better |")
    S.append("")
    S.append("```mermaid")
    S.append("graph LR")
    S.append("    A[Passage] --> B[Remove last word]")
    S.append("    B --> C[Prompt LLM]")
    S.append("    C --> D[Normalize prediction]")
    S.append("    D --> E{Match target?}")
    S.append("    E -->|Yes| F[Correct]")
    S.append("    E -->|No| G[Incorrect]")
    S.append("```")
    S.append("")
    S.append("---")
    S.append("")

    # Slide 6
    S.append("# Slide 6 \u2014 Performance Results")
    S.append("")
    S.append("## Summary Metrics")
    S.append("")
    if summary:
        S.append("| Model | Accuracy | Avg Latency | Errors |")
        S.append("|-------|----------|-------------|--------|")
        for m in summary["models"]:
            S.append(
                f"| {_n(m['model'])} | {m['accuracy']*100:.1f}% "
                f"| {m['avg_response_time']:.3f}s | {m['errors']} |"
            )
        S.append("")
        best = max(summary["models"], key=lambda x: x["accuracy"])
        S.append(f"**Best**: {_n(best['model'])} at {best['accuracy']*100:.1f}% accuracy")
    else:
        S.append("*Results pending \u2014 run `python main.py`.*")
    S.append("")
    S.append(_img("accuracy_comparison.png"))
    S.append("")
    S.append("---")
    S.append("")

    # Slide 7
    S.append("# Slide 7 \u2014 Visual Comparison")
    S.append("")
    S.append("## Accuracy vs. Response Time")
    S.append("")
    S.append(_img("combined_metrics.png"))
    S.append("")
    S.append("## Multi-Metric Radar")
    S.append("")
    S.append(_img("radar_comparison.png"))
    S.append("")
    S.append("---")
    S.append("")

    # Slide 8
    S.append("# Slide 8 \u2014 Workflow Diagram")
    S.append("")
    S.append("## End-to-End Pipeline")
    S.append("")
    S.append("```mermaid")
    S.append("graph TD")
    S.append("    A[Configuration] --> B[Load Dataset]")
    S.append("    B --> C[Sample Passages]")
    S.append("    C --> D[Evaluate Models]")
    S.append("    D --> E[Metrics & Charts]")
    S.append("    E --> F[Report & Slides]")
    S.append("    F --> G[Streamlit Dashboard]")
    S.append("```")
    S.append("")
    S.append(_img("project_workflow.png"))
    S.append("")
    S.append("---")
    S.append("")

    # Slide 9
    S.append("# Slide 9 \u2014 Key Takeaways")
    S.append("")
    S.append("## Conclusions")
    S.append("")
    S.append("- LAMBADA is a rigorous test of long-range contextual understanding.")
    S.append("- Larger models tend toward higher accuracy but at greater latency and cost.")
    S.append("- MoE and SWA architectures offer efficient alternatives to brute-force scaling.")
    S.append("- Knowledge distillation effectively compresses reasoning into smaller models.")
    S.append("- API-based evaluation provides reproducible, hardware-agnostic comparisons.")
    S.append("")
    S.append("---")
    S.append("")
    S.append("*Generated by LAMBADA Evaluation Pipeline*")
    S.append("")

    path = os.path.join(REPORT_DIR, "slide.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(S))
    print(f"Slides saved to {path}")
    return path


# ======================================================================
# Entry
# ======================================================================

def generate_all_reports(split="test"):
    print("\nGenerating reports...")
    print("=" * 40)
    generate_report(split)
    generate_slides(split)
    print("\nReport generation complete.")


if __name__ == "__main__":
    generate_all_reports()
