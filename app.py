"""
Streamlit Dashboard — LAMBADA Benchmark Evaluation

Interactive interface for exploring LLM evaluation results,
comparing model performance, and browsing the LAMBADA dataset.
"""

import os
import json

import streamlit as st
import streamlit.components.v1 as components

from config import MODELS, MODEL_INFO, DATASET_FILES, RESULTS_DIR, DIAGRAMS_DIR

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LAMBADA Benchmark Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _short(model_id):
    return MODEL_INFO.get(model_id, {}).get("name", model_id.split("/")[-1])


@st.cache_data
def load_summary(split="test"):
    path = os.path.join(RESULTS_DIR, f"summary_{split}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_model_results(model_id, split="test"):
    safe = model_id.replace("/", "_")
    path = os.path.join(RESULTS_DIR, f"{safe}_lambada_{split}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_dataset_lines(split, max_lines=500):
    path = DATASET_FILES.get(split)
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            lines.append(line.strip())
    return lines


def render_mermaid(code, height=420):
    html = f"""
    <div class="mermaid" style="display:flex;justify-content:center;padding:16px;">
    {code}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad:true, theme:'default'}});</script>
    """
    components.html(html, height=height, scrolling=True)


# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────

st.sidebar.title("📊 LAMBADA Eval")
st.sidebar.markdown("---")

model_names = {m: _short(m) for m in MODELS}
selected_model = st.sidebar.selectbox(
    "Select LLM",
    MODELS,
    format_func=lambda m: model_names[m],
)

split_options = list(DATASET_FILES.keys())
selected_split = st.sidebar.selectbox("Dataset Split", split_options, index=0)

benchmark_choice = st.sidebar.selectbox(
    "Benchmark", ["LAMBADA"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit • OpenRouter API")

# ──────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🏠 Overview",
    "🔬 Model Details",
    "📈 Benchmark Results",
    "⚖️ Comparison",
    "📂 Dataset Explorer",
    "🖼️ Diagrams",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 0 — Overview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[0]:
    st.header("LAMBADA Benchmark Evaluation")
    st.markdown(
        """
        This dashboard presents an interactive evaluation of **three Small Language Models**
        on the **LAMBADA** benchmark — a word-prediction task that tests long-range
        contextual understanding.

        **Models evaluated:**
        """
    )
    cols = st.columns(3)
    for i, m in enumerate(MODELS):
        info = MODEL_INFO.get(m, {})
        with cols[i]:
            st.metric(info.get("name", m), info.get("params", "N/A"))
            st.caption(info.get("developer", ""))

    st.markdown("### What is LAMBADA?")
    st.markdown(
        "The **LAMBADA** dataset evaluates a model's ability to predict the last word of "
        "a narrative passage. The passages are curated so that humans can guess the word "
        "when reading the full context, but **not** from the last sentence alone. This makes "
        "it a rigorous test of long-range dependency understanding."
    )

    st.markdown("### Project Workflow")
    render_mermaid(
        """
        graph TD
            A[Configuration] --> B[Load LAMBADA Dataset]
            B --> C[Sample Passages]
            C --> D{Evaluate Each Model}
            D --> E1[Grok-3-Mini]
            D --> E2[Ministral-14B]
            D --> E3[DeepSeek-R1-Distill]
            E1 --> F[Collect Predictions]
            E2 --> F
            E3 --> F
            F --> G[Calculate Metrics]
            G --> H[Generate Report]
            H --> I[Dashboard]
        """,
        height=500,
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Model Details
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[1]:
    info = MODEL_INFO.get(selected_model, {})
    st.header(f"🔬 {info.get('name', selected_model)}")
    st.markdown(f"**Developer:** {info.get('developer', 'N/A')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Parameters:** {info.get('params', 'N/A')}")
    with col2:
        st.markdown(f"**Architecture:** {info.get('architecture', 'N/A')}")

    st.markdown("### Working Principle")
    st.markdown(info.get("description", "No description available."))

    st.markdown("### Strengths & Weaknesses")
    scol1, scol2 = st.columns(2)
    with scol1:
        st.success(f"**Strengths:** {info.get('strengths', 'N/A')}")
    with scol2:
        st.warning(f"**Weaknesses:** {info.get('weaknesses', 'N/A')}")

    st.markdown("### All Models at a Glance")
    table_data = []
    for m in MODELS:
        mi = MODEL_INFO.get(m, {})
        table_data.append(
            {
                "Model": mi.get("name", m),
                "Developer": mi.get("developer", "N/A"),
                "Parameters": mi.get("params", "N/A"),
                "Architecture": mi.get("architecture", "N/A"),
                "Strengths": mi.get("strengths", "N/A"),
            }
        )
    st.table(table_data)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Benchmark Results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[2]:
    st.header(f"📈 {benchmark_choice} Results — {_short(selected_model)}")

    summary = load_summary(selected_split)
    detail = load_model_results(selected_model, selected_split)

    if summary:
        model_entry = next(
            (m for m in summary["models"] if m["model"] == selected_model), None
        )
        if model_entry:
            st.markdown(f"**Split:** {selected_split} &nbsp;|&nbsp; **Samples:** {summary['num_samples']}")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Accuracy", f"{model_entry['accuracy']*100:.1f}%")
            mcol2.metric("Correct", model_entry["correct"])
            mcol3.metric("Avg Time", f"{model_entry['avg_response_time']:.3f}s")
            mcol4.metric("Errors", model_entry["errors"])
        else:
            st.info(f"No results for {_short(selected_model)} on {selected_split} split.")
    else:
        st.warning(
            f"No summary results found for **{selected_split}** split. "
            "Run `python main.py` first."
        )

    if detail and detail.get("results"):
        st.markdown("### Per-sample Results")
        sample_data = []
        for r in detail["results"]:
            sample_data.append(
                {
                    "#": r["index"],
                    "Target": r["target"],
                    "Prediction": r["prediction"],
                    "Correct": "✅" if r["correct"] else "❌",
                    "Time (s)": r["time"],
                }
            )
        st.dataframe(sample_data, use_container_width=True, height=400)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[3]:
    st.header("⚖️ Model Comparison")
    summary = load_summary(selected_split)

    if summary:
        st.markdown(f"**Split:** {selected_split} &nbsp;|&nbsp; **Samples:** {summary['num_samples']}")

        comp_data = []
        for m in summary["models"]:
            comp_data.append(
                {
                    "Model": _short(m["model"]),
                    "Accuracy (%)": f"{m['accuracy']*100:.1f}",
                    "Correct": m["correct"],
                    "Total": m["total"],
                    "Avg Time (s)": f"{m['avg_response_time']:.3f}",
                    "Errors": m["errors"],
                }
            )
        st.table(comp_data)

        st.markdown("### Accuracy Comparison")
        acc_path = os.path.join(DIAGRAMS_DIR, "accuracy_comparison.png")
        if os.path.exists(acc_path):
            st.image(acc_path, use_container_width=True)
        else:
            accs = {_short(m["model"]): m["accuracy"] * 100 for m in summary["models"]}
            st.bar_chart(accs)

        st.markdown("### Response Time Comparison")
        rt_path = os.path.join(DIAGRAMS_DIR, "response_time.png")
        if os.path.exists(rt_path):
            st.image(rt_path, use_container_width=True)
        else:
            rts = {_short(m["model"]): m["avg_response_time"] for m in summary["models"]}
            st.bar_chart(rts)

        st.markdown("### Combined Metrics")
        cb_path = os.path.join(DIAGRAMS_DIR, "combined_metrics.png")
        if os.path.exists(cb_path):
            st.image(cb_path, use_container_width=True)
    else:
        st.warning("No results available. Run `python main.py` first.")

    st.markdown("### Evaluation Architecture")
    render_mermaid(
        """
        graph LR
            subgraph Input
                A[LAMBADA Passage] --> B[Remove Last Word]
                B --> C[Context Prompt]
            end
            subgraph OpenRouter_API
                C --> D[Grok-3-Mini]
                C --> E[Ministral-14B]
                C --> F[DeepSeek-R1-Distill]
            end
            subgraph Evaluation
                D --> G[Prediction]
                E --> G
                F --> G
                G --> H{Exact Match?}
                H --> I[Accuracy Score]
            end
        """,
        height=350,
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — Dataset Explorer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[4]:
    st.header("📂 Dataset Explorer")

    ds_split = st.selectbox(
        "Choose split to browse",
        list(DATASET_FILES.keys()),
        key="ds_explorer_split",
    )

    lines = load_dataset_lines(ds_split)
    if lines:
        st.markdown(f"Showing up to **{len(lines)}** passages from **{ds_split}** split.")

        st.markdown("#### Dataset Statistics")
        ds_col1, ds_col2, ds_col3 = st.columns(3)
        ds_col1.metric("Passages loaded", len(lines))
        avg_len = sum(len(l.split()) for l in lines) / len(lines) if lines else 0
        ds_col2.metric("Avg words / passage", f"{avg_len:.0f}")
        targets = [l.rsplit(" ", 1)[-1] if " " in l else "" for l in lines]
        unique_targets = len(set(targets))
        ds_col3.metric("Unique target words", unique_targets)

        st.markdown("#### Sample Passages")
        page_size = 20
        total_pages = max(1, (len(lines) + page_size - 1) // page_size)
        page = st.number_input("Page", 1, total_pages, 1, key="ds_page")
        start = (page - 1) * page_size
        end = min(start + page_size, len(lines))

        for idx in range(start, end):
            passage = lines[idx]
            words = passage.rsplit(" ", 1)
            if len(words) == 2:
                context, target = words
                st.markdown(
                    f"**#{idx + 1}** &nbsp; {context} **`{target}`**"
                )
            else:
                st.markdown(f"**#{idx + 1}** &nbsp; {passage}")
    else:
        st.warning(f"No data found for split **{ds_split}**.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — Diagrams
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tabs[5]:
    st.header("🖼️ Generated Diagrams")

    diagram_files = [
        ("Project Workflow", "project_workflow.png"),
        ("Architecture Comparison", "architecture_comparison.png"),
        ("Evaluation Pipeline", "evaluation_pipeline.png"),
        ("LAMBADA Task", "lambada_task.png"),
        ("Accuracy Comparison", "accuracy_comparison.png"),
        ("Response Time", "response_time.png"),
        ("Combined Metrics", "combined_metrics.png"),
    ]

    found_any = False
    for title, fname in diagram_files:
        path = os.path.join(DIAGRAMS_DIR, fname)
        if os.path.exists(path):
            found_any = True
            st.markdown(f"### {title}")
            st.image(path, use_container_width=True)

    if not found_any:
        st.info(
            "No diagrams found. Run `python main.py` or "
            "`python generate_diagrams.py` to generate them."
        )

    st.markdown("### Live Mermaid: Evaluation Pipeline")
    render_mermaid(
        """
        graph TD
            A[LAMBADA Dataset] --> B[Test: 5153 passages]
            A --> C[Dev: 4869 passages]
            A --> D[Rejected Data]
            B --> E[Random Sample]
            C --> E
            E --> F[Extract Context + Target]
            F --> G[Construct Prompt]
            G --> H[OpenRouter API Call]
            H --> I[Parse Response]
            I --> J[Normalize]
            J --> K[Exact Match]
            K --> L[Accuracy]
            K --> M[Per-sample Results]
            H --> N[Latency]
            L --> O[Summary JSON]
            M --> O
            N --> O
        """,
        height=600,
    )
